# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import checkpoint_utils, utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.voctransformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.voctransformer.transformer_base import (
    TransformerModelBase,
)
from pathlib import Path
import torch

@register_model("voctransformer")
class VOCTransformerModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-convpre-from",
            type=str,
            metavar="STR",
            help="vocoder to take conv-pre weights from (for initialization)",
        )
        

        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, units_embed_dim,  path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, units_embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = super().build_encoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )


        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                print(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                print(f"loaded pretrained encoder from: {pretraining_path}")
        

        vocoder_path = getattr(args, "load_convpre_from", None)
        if vocoder_path is not None:
            if not Path(vocoder_path).exists():
                print(
                    f"skipping initializing with vocoder because {vocoder_path} doen not exist"
                )
            else:
                vocoder_state = torch.load(vocoder_path)

                new_state_dict = encoder.state_dict()
                new_state_dict["conv_pre.bias"] = vocoder_state["generator"]["conv_pre.bias"]
                new_state_dict["conv_pre.weight_g"] = vocoder_state["generator"]["conv_pre.weight_g"]
                new_state_dict["conv_pre.weight_v"] = vocoder_state["generator"]["conv_pre.weight_v"]
                encoder.load_state_dict(new_state_dict)

                print(f"loaded pretrained conv_pre from: {vocoder_path}")

        if args.freeze_emb:
            encoder.conv_pre.bias.requires_grad = False
            encoder.conv_pre.weight_g.requires_grad = False
            encoder.conv_pre.weight_v.requires_grad = False
            print('freezing conv_pre parameters')

        return encoder

        
    
#         return super().build_encoder(
#             TransformerConfig.from_namespace(args), src_dict, embed_tokens
#         )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        
        decoder = super().build_decoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None and pretraining_path != "":
            if not Path(pretraining_path).exists():
                print(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                print(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

        # return super().build_decoder(
        #         TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        #     )
# architectures




@register_model_architecture("voctransformer", "voctransformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.units_embed_dim = getattr(args, "units_embed_dim", 128)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

