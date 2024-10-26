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
from argparse import ArgumentParser

class SpeechDataset(torch.utils.data.Dataset):
    value2class = {'1.0':0, '1.5':1, '2.0':2, '2.5':3, '3.0':4, '3.5':5, '4.0':6, '4.5':7, '5.0':8}

    def __init__(self, csv_path:str, target_speaker:str, sample_rate=16000, train_df=None, loss_type='mse') -> None:
        super().__init__()

        self.df = None
        self.sample_rate = sample_rate

        df = pd.read_csv(csv_path)
        self.mean = df['intelligibility'].mean()
        self.std = df['intelligibility'].std()
        
        if train_df is None:
            self.df = pd.read_csv(csv_path).query('speaker!=@target_speaker').sample(frac=0.90)
        else:
            self.df = pd.read_csv(csv_path).query('speaker!=@target_speaker')
            self.df = self.df.merge(train_df, on=['path', 'speaker', 'intelligibility'], how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

        self.loss_type = loss_type

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
        value=0
        if self.loss_type == 'kappa':
            value = SpeechDataset.value2class[row['intelligibility']]
        else:
            value = (float(row['intelligibility']) - self.mean)/self.std

        try:
            # torchaudioで読み込んだ場合，音声データはFloatTensorで（チャンネル，サンプル数）
            wave, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate, dtype=wave.dtype)
                wave = resampler(wave)
            # 平均をゼロ，分散を1に正規化
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std
            length = wave.shape[-1]
        except:
            raise RuntimeError('file open error')
               
        return wave, length, value, self.loss_type

'''
    バッチデータの作成
'''
def data_processing(data:Tuple[Tensor,int]) -> Tuple[Tensor, Tensor]:
    waves = []
    lengths = []
    values = []

    loss_type = 'kappa'
    for wave, length, value, _type in data:
        # w/ channel
        waves.append(wave.t())
        lengths.append(length//320-1) # wav2vec2 outout length
        loss_type=_type
        values.append(value)

    # データはサンプル数（長さ）が異なるので，長さを揃える
    # 一番長いサンプルよりも短いサンプルに対してゼロ詰めで長さをあわせる
    # バッチはFloatTensorで（バッチサイズ，チャンネル，サンプル数）
    waves = nn.utils.rnn.pad_sequence(waves, batch_first=True)
    waves = rearrange(waves, 'b t c -> b (t c)')
    #lengths = torch.tensor(lengths)
    #packed = nn.utils.rnn.pack_padded_sequence(waves.unsqueeze(-1).float(), lengths, batch_first=True, enforce_sorted=False)

    # 話者のインデックスを配列（Tensor）に変換
    if loss_type == 'kappa':
        values = torch.from_numpy(np.array(values)).clone().int()
    else:
        values = torch.from_numpy(np.array(intelligibilities)).clone().float()

    return waves, None, values

if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--target_speaker', type=str, default='F004')
    args=parser.parse_args()

    train_dataset = SpeechDataset(args.csv, args.target_speaker)
    train_df = train_dataset.get_df()
    valid_dataset = SpeechDataset(args.csv, args.target_speaker, train_df=train_df)
    print(train_dataset.__len__())
    print(valid_dataset.__len__())
    
    wave, intelligibility = train_dataset.__getitem__(10)
