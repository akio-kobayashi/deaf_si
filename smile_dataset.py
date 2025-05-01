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
from einops import rearrange
import coral_loss

class SmileDataset(torch.utils.data.Dataset):

    def __init__(self, path:str, stat_path:str) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        stats = torch.load(stat_path)
        self.smile_mean = stats['smile_mean'].to(self.device) # (1, 88)
        self.smile_var = stats['smile_var'].to(self.device)   # (1, 88)
        self.mfcc_mean = rearrange(stats['mfcc_mean'].to(self.device), '(b t) -> b t', t=1)   # (40)
        self.mfcc_var = rearrange(stats['mfcc_var'].to(self.device), '(b t) -> b t', t=1)     # (40)
        self.df = pd.read_csv(path)
        
        self.data_length = len(self.df)
        self.score2rank = {'1.0':1, '1.5':2, '2.0':3, '2.5':4, '3.0':5, '3.5':6, '4.0':7, '4.5':8, '5.0':9}
        
    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]

        data = torch.load(row['feature'])
        smile = data['smile'].to(self.device) # (1, 88)
        mfcc = data['mfcc'].to(self.device)   # (40, T)
        
        mfcc = (mfcc - self.mfcc_mean)/self.mfcc_var
        smile = (smile - self.smile_mean)/self.smile_var

        rank = self.score2rank[row['intelligibility'].astype(str)]

        return mfcc, smile, rank
    
def data_processing(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mfcc, smile, rank, lengths = [], [], [], []

    for _m, _s, _r in data:
        mfcc.append(rearrange(_m, 'f t -> t f'))
        lengths.append(_m.shape[-1])
        smile.append(_s)
        rank.append(_r)
    rank = torch.from_numpy(np.array(rank)).to(device)
    lengths = torch.from_numpy(np.array(lengths)).to(device)
    labels = coral_loss.ordinal_labels(rank, 9)
    mfcc = nn.utils.rnn.pad_sequence(mfcc, batch_first=True)
    smile = rearrange(nn.utils.rnn.pad_sequence(smile, batch_first=True), 'b c f -> (b c) f')

    return mfcc, smile, labels, rank, lengths

