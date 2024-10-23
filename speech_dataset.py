import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple, List
from torch import Tensor
import torchaudio
import scipy.signal as signal
from einops import rearrange

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, target_speaker:str, sample_rate=16000, train_df=None) -> None:
        super().__init__()

        self.df = None
        self.sample_rate = sample_rate

        if train_df is None:
            self.df = pd.read_csv(csv_path).query('speaker!=@target_speaker').sample(frac=0.90)
        else:
            self.df = pd.read_csv(csv_path).query('speaker!=@target_speaker')
            self.df = self.df.merge(train_df, on=['path', 'speaker', 'intelligibility'], how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    def get_df(self):
        return self.df
    
    def __len__(self) -> int:
        return len(self.df)

    '''
        データフレームからidx番目のサンプルを抽出する
    '''
    def __getitem__(self, idx:int) -> Tuple[Tensor, int]:
        row = self.df.iloc[idx]
        # audio path
        path = row['path']
        intelligiblity = float(row['intelligibility'])

        try:
            # torchaudioで読み込んだ場合，音声データはFloatTensorで（チャンネル，サンプル数）
            wave, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.Resample(sr, self.sample_rate, dtype=wave.dtype)
                wave = resampler(wave)
            # 平均をゼロ，分散を1に正規化
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std            
        except:
            raise RuntimeError('file open error')
               
        return wave, intelligiblity

'''
    バッチデータの作成
'''
def data_processing(data:Tuple[Tensor,int]) -> Tuple[Tensor, Tensor]:
    waves = []
    intelligibilities = []

    for wave, intelligibility in data:
        # w/ channel
        waves.append(wave)
        intelligibilities.append(intelligibility)

    # データはサンプル数（長さ）が異なるので，長さを揃える
    # 一番長いサンプルよりも短いサンプルに対してゼロ詰めで長さをあわせる
    # バッチはFloatTensorで（バッチサイズ，チャンネル，サンプル数）
    waves = nn.utils.rnn.pad_sequence(waves, batch_first=True)
    waves = rearrange(waves, 'b c t -> b (c t)')
    
    # 話者のインデックスを配列（Tensor）に変換
    intelligibilities = torch.from_numpy(np.array(intelligibilities)).clone()

    return waves, intelligibilities
