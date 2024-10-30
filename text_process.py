import numpy as np
import sys, os, re, gzip, struct
import random
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.utils.data as data
#import pandas as pd
from typing import Tuple, List
#from torch import Tensor
#import torchaudio
#import scipy.signal as signal
#from einops import rearrange
from argparse import ArgumentParser

class TextProcessor():
    def __init__(self, path):
        super().__init__()
        self.phone2id, self.id2phone = {}, {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                phone, num = line.strip().split(' ')
                self.phone2id[phone] = int(num)
                self.id2phone[int(num)] = phone
                
    def to_seq(self, text):
        token = text.strip().split(' ')
        if token[0] not in self.phone2id.keys():
            del token[0]
        if token[0] != '<bos>':
            token.insert(0, '<bos>')
        if token[-1] != '<eos>':
            token.append('<eos>')
        try:
            seq = [ self.phone2id[t] for t in token ]
            return seq
        except:
            print(f'token error {text}')
    
    def to_text(self, seq):
        token = [ self.id2phone[num] for num in seq ]
        token.remove('<bos>')
        token.remove('<eos>')
        return ' '.join(token)

if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    args=parser.parse_args()

    processor = TextProcessor(args.vocab)

    with open(args.text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            seq = processor.to_seq(line)
            txt = processor.to_text(seq)
            print(txt)
            
