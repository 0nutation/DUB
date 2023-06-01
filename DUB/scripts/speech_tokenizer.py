from genericpath import exists
from typing import List, Union
import logging
import os
import sys

import joblib
import fire
import fairseq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import re
import numpy as np
from functools import partial
import torch.multiprocessing as mp
import torchaudio
import glob
import tqdm
import argparse

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('generate_pseudo_language')


class FeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, pool_k=1, pool_s=1):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.encoder_type = 'hubert'

        if pool_k == 1 and pool_s == 1:
            self.pooler = None
        else:
            self.pooler = nn.AvgPool1d(pool_k, stride=pool_s)
        
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
            
                if self.pooler is not None:
                    feat_chunk = rearrange(feat_chunk, 'b t c -> b c t')
                    feat_chunk = self.pooler(feat_chunk)
                    feat_chunk = rearrange(feat_chunk, 'b c t -> b t c')
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)




class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class SpeechTokenizer(torch.nn.Module):
    def __init__(
        self, 
        ckpt_path, 
        layer, 
        max_chunk, 
        fp16, 
        pool_k, 
        pool_s, 
        km_path,
        ):

        """
        Args:
            ckpt_path(str): path to hubert model(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            pool_k(int): pooling_kernal default by 1
            pool_s(int): pooling_stride default by 1
            km_path: path to pretrained kmeans model(e.g. c500_km.pkl)
        """
        super().__init__()
        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, pool_k, pool_s)
        self.apply_kmeans = ApplyKmeans(km_path)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list
    

    def __call__(self, waveform):
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
        return {"continuous":feat, "units":dup_cluster_list, "duration":duration_list, "unmerged_units":cluster_ids}



    
