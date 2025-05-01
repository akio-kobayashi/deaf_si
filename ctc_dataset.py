import numpy as np
import time
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
from speech_dataset import SpeechDataset
from text_process import TextProcessor

class CTCDataset(SpeechDataset):
    value2class = {'1.0':0, '1.5':1, '2.0':2, '2.5':3, '3.0':4, '3.5':5, '4.0':6, '4.5':7, '5.0':8}

    def __init__(self, csv_path:str, ctc_path:str, vocab_path:str, text_path:str, target_speaker:str, sample_rate=16000,
                 train_df=None, train_ctc_df=None, loss_type='mse', return_length=True, sort_by_length=True) -> None:
        super().__init__(csv_path, target_speaker, sample_rate, train_df, loss_type, return_length)

        self.total_ctc_df = None
        if train_ctc_df is None:
            self.total_ctc_df = pd.read_csv(ctc_path).query('speaker!=@target_speaker').sample(frac=0.80)
        else:
            self.total_ctc_df = pd.read_csv(ctc_path).query('speaker!=@target_speaker')
            self.total_ctc_df = self.total_ctc_df.merge(train_ctc_df, on=['path', 'speaker', 'atr', 'length'],
                                                        how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            self.total_ctc_df = self.total_ctc_df.sample(len(self.df), random_state=np.random.randint(1000, dtype='int32'))
        #self.ctc_df = self.total_ctc_df.sample(len(self.df), random_state=np.random.randint(1000, dtype='int32'))
        self.processor = TextProcessor(vocab_path)
        self.atr_text = {}
        with open(text_path, 'r') as f:
            for line in f:
                tag = line.split()[0]
                text = ' '.join(line.strip().split()[1:])
                self.atr_text[tag] = text
        # if process sorted order
        self.ctc_idx = 0
        self.sort_by_length = sort_by_length
        if self.sort_by_length:
            self.total_ctc_df = self.total_ctc_df.sort_values(by='length')

    def get_ctc_df(self):
        return self.total_ctc_df

    '''
        CTCのデータフレームをリセットし，ランダムにサンプルを取得
    '''    
    def reset_ctc_df(self):
        #self.ctc_df = self.total_ctc_df.sample(len(self.df), random_state=np.random.randint(1000, dtype='int32'))
        self.ctc_idx = 0
        #if self.sort_by_length:
        #    self.ctc_df = self.ctc_df.sort_values(by='length')

    '''
        データフレームからidx番目のサンプルを抽出する
    '''
    def __getitem__(self, idx:int) -> Tuple[Tensor, int]:
        wave_si, _, value, self.loss_type = super().__getitem__(idx)

        #row = self.ctc_df.iloc[idx]
        #if self.sort_by_length:
        if self.ctc_idx >= len(self.total_ctc_df):
            sefl.ctc_idx = 0
        #row = self.ctc_df.iloc[self.ctc_idx]
        row = self.total_ctc_df.iloc[self.ctc_idx]
        self.ctc_idx += 1
        
        wave_ctc_path = row['path']
        wave_ctc, sr = torchaudio.load(wave_ctc_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate, dtype=wave.dtype)
            wave_ctc = resampler(wave_ctc)
        # 平均をゼロ，分散を1に正規化
        std, mean = torch.std_mean(wave_ctc, dim=-1)
        wave_ctc = (wave_ctc - mean)/std
        label = self.processor.to_seq(self.atr_text[row['atr']])
        
        return wave_si, wave_ctc, label, value, self.loss_type

'''
    バッチデータの作成
'''
def data_processing(data:Tuple[Tensor, list, int, float, str]) -> Tuple[Tensor, Tensor, Tensor, list, list, Tensor]:
    waves_si = []
    waves_ctc = []
    lengths_si = []
    lengths_ctc = []
    values = []
    labels = []
    label_lengths = []
    loss_type = 'kappa'
    for wave, wave_ctc, label, value, _type in data:
        # w/ channel
        lengths_si.append(wave.shape[-1]//320-1) # wav2vec2 outout length
        lengths_ctc.append(wave_ctc.shape[-1]//320 -1)
        waves_si.append(wave.t())
        waves_ctc.append(wave_ctc.t())
        loss_type=_type
        values.append(value)
        label_lengths.append(len(label))
        labels.append(torch.from_numpy(np.array(label)))

    waves_si = nn.utils.rnn.pad_sequence(waves_si, batch_first=True)
    waves_si = rearrange(waves_si, 'b t c -> b (t c)')
    waves_ctc = nn.utils.rnn.pad_sequence(waves_ctc, batch_first=True)
    waves_ctc = rearrange(waves_ctc, 'b t c -> b (t c)')
    
    lengths_si_tensor = torch.tensor(lengths_si).to(torch.int64)
    lengths_ctc_tensor = torch.tensor(lengths_ctc).to(torch.int64)
    
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    label_lengths_tensor = torch.tensor(label_lengths).to(torch.int64)
    
    # 話者のインデックスを配列（Tensor）に変換
    if loss_type == 'kappa' or loss_type == 'custom':
        values = torch.from_numpy(np.array(values)).clone().to(torch.int64)
    else:
        values = torch.from_numpy(np.array(values)).clone().float()

    return (waves_si, values, lengths_si_tensor, lengths_si), (waves_ctc, labels, lengths_ctc_tensor, lengths_ctc, label_lengths_tensor, label_lengths)

if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--target_speaker', type=str, default='F004')
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    
    train_dataset = CTCDataset(config['csv'], config['ctc_csv'], config['vocab'], config['text'],
                               args.target_speaker)
    train_df = train_dataset.get_df()
    valid_dataset = CTCDataset(config['csv'], config['ctc_csv'], config['vocab'], config['text'],
                               args.target_speaker, train_df=train_df)
    print(train_dataset.__len__())
    print(valid_dataset.__len__())

    wave, wave_ctc, label, value, loss_type = train_dataset.__getitem__(10)
