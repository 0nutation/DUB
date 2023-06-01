# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .Seq2SeqHubert_config import (
    Seq2SeqHubertConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
from .transformer_encoder import TransformerEncoder, TransformerEncoderBase
from .Seq2SeqHubert_legacy import (
    Seq2SeqHubertModel,
    base_architecture,
    tiny_architecture,
    Seq2SeqHubert_iwslt_de_en,
    Seq2SeqHubert_wmt_en_de,
    Seq2SeqHubert_vaswani_wmt_en_de_big,
    Seq2SeqHubert_vaswani_wmt_en_fr_big,
    Seq2SeqHubert_wmt_en_de_big,
    Seq2SeqHubert_wmt_en_de_big_t2t,
)
from .Seq2SeqHubert_base import Seq2SeqHubertModelBase, Embedding
from .hubert_encoder import Seq2SeqHubertEncoder

__all__ = [
    "Seq2SeqHubertModelBase",
    "Seq2SeqHubertConfig",
    "TransformerDecoder",
    "TransformerDecoderBase",
    "Seq2SeqHubertEncoder",
    "TransformerEncoder",
    "TransformerEncoderBase",
    "Seq2SeqHubertModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "Seq2SeqHubert_iwslt_de_en",
    "Seq2SeqHubert_wmt_en_de",
    "Seq2SeqHubert_vaswani_wmt_en_de_big",
    "Seq2SeqHubert_vaswani_wmt_en_fr_big",
    "Seq2SeqHubert_wmt_en_de_big",
    "Seq2SeqHubert_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
    
]
