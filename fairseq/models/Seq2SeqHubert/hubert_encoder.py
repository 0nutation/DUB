# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from fairseq.modules import GradMultiply, PositionalEmbedding, TransposeLast, SamePad
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.Seq2SeqHubert import (
    Seq2SeqHubertConfig,
)
from fairseq.models.wav2vec import pad_to_multiple, TransformerSentenceEncoderLayer, make_conv_pos
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
)
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
import numpy as np


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class Seq2SeqHubertEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, post_extract_proj,return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        word_embed_dim = embed_tokens.embedding_dim
        enc_embed_dim = cfg.encoder.embed_dim


        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(word_embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                word_embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )



        self.layer_norm = LayerNorm(word_embed_dim)

        # self.post_extract_proj = (
        #     nn.Linear(word_embed_dim, enc_embed_dim)
        #     if word_embed_dim != enc_embed_dim
        #     else None
        # )
        self.post_extract_proj = post_extract_proj

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(word_embed_dim, word_embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        # if self.encoder_layerdrop > 0.0:
        #     self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        # else:
        #     self.layers = nn.ModuleList([])
        # self.layers.extend(
        #     [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        # )
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        # final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )


        self.encoder = TransformerEncoder(cfg)
        self.num_layers = len(self.encoder.layers)



    # def build_encoder_layer(self, cfg):
    #     layer = transformer_layer.TransformerEncoderLayerBase(
    #         cfg, return_fc=self.return_fc
    #     )
    #     checkpoint = cfg.checkpoint_activations
    #     if checkpoint:
    #         offload_to_cpu = cfg.offload_activations
    #         layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
    #     # if we are checkpointing, enforce that FSDP always wraps the
    #     # checkpointed layer, regardless of layer size
    #     min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
    #     layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
    #     return layer

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices


    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        return_all_layers: bool = False
    ) :

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = encoder_padding_mask.any()
        features = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            features = features + self.embed_positions(src_tokens)
        if has_pads:
            features = features * (1 - encoder_padding_mask.unsqueeze(-1).type_as(features))

        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)



        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)


        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None


        # x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return {
            "encoder_out": [x],   # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "features": [features], # B x Tx C
                }



    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]


        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]


        if len(encoder_out["features"]) == 0:
            new_features = []
        else:
            new_features = [
                encoder_out["features"][0].index_select(0, new_order)
            ]


        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "features": new_features,  # B x T x C
        }


    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
    #         weights_key = "{}.embed_positions.weights".format(name)
    #         if weights_key in state_dict:
    #             print("deleting {0}".format(weights_key))
    #             del state_dict[weights_key]
    #         state_dict[
    #             "{}.embed_positions._float_tensor".format(name)
    #         ] = torch.FloatTensor(1)
    #     for i in range(self.num_layers):
    #         # update layer norms
    #         self.layers[i].upgrade_state_dict_named(
    #             state_dict, "{}.layers.{}".format(name, i)
    #         )

    #     version_key = "{}.version".format(name)
    #     if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
    #         # earlier checkpoints did not normalize after the stack of layers
    #         self.layer_norm = None
    #         self.normalize = False
    #         state_dict[version_key] = torch.Tensor([1])
    #     return state_dict



class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, cfg):
        super().__init__()

        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder.embed_dim
        self.required_seq_len_multiple = cfg.required_seq_len_multiple

        if getattr(cfg, "no_pos_conv", False):
            self.pos_conv = None
        else:
            pos_conv_depth = getattr(cfg, "pos_conv_depth", 1)
            if pos_conv_depth > 1:
                num_layers = cfg.pos_conv_depth
                k = max(3, cfg.conv_pos // num_layers)

                def make_conv_block(e, k, g, l):
                    return nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.Conv1d(
                                    e,
                                    e,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=g,
                                ),
                                SamePad(k),
                                TransposeLast(),
                                LayerNorm(e, elementwise_affine=False),
                                TransposeLast(),
                                nn.GELU(),
                            )
                            for _ in range(l)
                        ]
                    )

                self.pos_conv = make_conv_block(
                    self.embedding_dim, k, cfg.conv_pos_groups, num_layers
                )

            else:
                self.pos_conv = make_conv_pos(
                    self.embedding_dim,
                    cfg.conv_pos,
                    cfg.conv_pos_groups,
                )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(cfg) for _ in range(cfg.encoder.layers)]
        )
        self.layer_norm_first = cfg.layer_norm_first
        # self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = cfg.encoder.layerdrop

        self.max_positions = cfg.max_source_positions
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        # if self.layer_norm_first and layer is None:
        #     x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
            self,
            x,
            padding_mask=None,
            tgt_layer=None,
            min_layer=0,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        # if not self.layer_norm_first:
        #     x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
