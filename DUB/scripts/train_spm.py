#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import sentencepiece as sp
import argparse


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_vocab(input_path: str,
    output_path_prefix: str,
    model_type="bpe",
    vocab_size=1000,
    accept_language=None,
    user_defined_symbols=None
    ):
    # Train SentencePiece Model
    sp.SentencePieceTrainer.train(input=input_path,
                                model_prefix=output_path_prefix,
                                model_type=model_type,
                                vocab_size=vocab_size,
                                accept_language=accept_language,
                                user_defined_symbols=user_defined_symbols,
                                unk_piece=UNK_TOKEN, unk_id=UNK_TOKEN_ID,
                                bos_piece=BOS_TOKEN, bos_id=BOS_TOKEN_ID,
                                eos_piece=EOS_TOKEN, eos_id=EOS_TOKEN_ID,
                                pad_piece=PAD_TOKEN, pad_id=PAD_TOKEN_ID,
                                add_dummy_prefix=False,
                                )
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")



if __name__ == "__main__":
    # spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--model-prefix", default="", type=str)
    parser.add_argument("--model-type", default="bpe", type=str)
    parser.add_argument("--user-defined-symbols", default="[]", type=str)
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--units-size", default=1000, type=int)
    parser.add_argument("--lang", default="en_units de", type=str)
    parser.add_argument("--units-prefix", default="#0", type=str)


    args = parser.parse_args()


    user_defined_symbols = []
    for lang in args.lang.split():
        user_defined_symbols += [f"<lang:{lang}>"]
    user_defined_symbols += [args.units_prefix.replace('0',str(i)) for i in range(args.units_size)] 
    user_defined_symbols += ['<BT>']
    user_defined_symbols += eval(args.user_defined_symbols)

    

    gen_vocab(
        input_path=args.input_path,
        output_path_prefix=args.model_prefix,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        user_defined_symbols=user_defined_symbols,
    )
    