#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import os
from itertools import groupby
from typing import BinaryIO, List, Optional, Tuple, Union

import pandas as pd
import torchaudio

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm
import soundfile as sf
from speech_tokenizer import SpeechTokenizer
from fairseq.data.audio.audio_utils import get_waveform, convert_waveform
import csv
import numpy as np

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "units", "src_text", "tgt_text", "speaker"]



class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


        


def process(args):
    root = Path(args.data_root).absolute()

    lang = args.language

    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    
    if not Path(args.hubert_dir).is_dir():
        os.makedirs(args.hubert_dir, exist_ok=True)
        os.system(f'wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O {args.hubert_dir}/hubert_model.pt')
        os.system(f'wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin -O {args.hubert_dir}/km.model')
    
    hubert_path = f'{args.hubert_dir}/hubert_model.pt'
    km_path = f'{args.hubert_dir}/km.model'

    speechtokenizer = SpeechTokenizer(
        ckpt_path=hubert_path,
        layer=9,
        max_chunk=1600000,
        fp16=False,
        pool_k=1,
        pool_s=1,
        km_path=km_path
    )

    def extract_discrete_units(
        waveform, #torch.Tensor
        types, # ["units", "unmerged_units", "duration", "continuous"]
        output_path=None,
        wav_name=None,
    ):
        
        _waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
        _waveform = _waveform.squeeze()

        encoded_audio = speechtokenizer(_waveform)
        units = encoded_audio[types]
        _units = ''.join(['#'+str(x) for x in units])

        if output_path is not None:
            with open(output_path,'a') as f:
                f.writelines(f'{wav_name}|{_units}')
        return _units

    

    for split in MUSTC.SPLITS:
        print(f"Fetching split {split}...")
        dataset = MUSTC(root.as_posix(), lang, split)
        with open(os.path.join(args.output_dir,f'{split}.id'),'a') as fid, \
             open(os.path.join(args.output_dir,f'{split}.en'),'a') as fs, \
             open(os.path.join(args.output_dir,f'{split}.{args.language}'),'a') as ft, \
             open(os.path.join(args.output_dir,f'{split}.en_units'),'a') as funits:

            for waveform, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                fid.writelines(utt_id + '\n')
                fs.writelines(src_utt + '\n')
                ft.writelines(tgt_utt + '\n')
                units = extract_discrete_units(waveform, "units")
                funits.writelines(units + '\n')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--hubert-dir", required=True, type=str,
        help="dirname of hubert model and kmeans model"
    )
    parser.add_argument("--language", required=True, type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()


    process(args)


if __name__ == "__main__":
    main()
