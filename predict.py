import matplotlib.pyplot as plt
import os, sys
import torch
import torchaudio
import pytorch_lightning as pl
import numpy as np
from solver import LightningSolver
from argparse import ArgumentParser
import pandas as pd
import yaml
import pickle
import pprint
import warnings
from einops import rearrange
warnings.filterwarnings('ignore')

def predict(config:dict, target_speaker, output_csv):

    lite = LightningSolver.load_from_checkpoint(config['checkpoint_path'], strict=False, config=config).cuda()
    lite.eval()

    predicts, targets = [], []
    files, intelligiblities = [], [] 
    with torch.no_grad():
        df = pd.read_csv(config['csv']).query('speaker==@target_speaker')
        for idx, row in df.iterrows():
            wave, sr = torchaudio.load(row['path'])
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std
            wave = rearrange(wave, '(b c) t -> b c t', b=1)
            pred = lite.forward(wave.cuda())
            predicts.append(pred.item())
            targets.append(float(row['intelligibility']))

            files.append(row['path'])
            
    output_df = pd.DataFrame.from_dict({'path': files, 'intelligibility': intelligiblities})
    output_df.to_csv(output_csv)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--target_speaker', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--gpu',  type=int, default=0)
    args=parser.parse_args()
    #torch.set_default_device("cuda:"+str(args.gpu))
    
    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config['checkpoint_path'] = args.ckpt
    predict(config, args.target_speaker, args.output_csv) 
