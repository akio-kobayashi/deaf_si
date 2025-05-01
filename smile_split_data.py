import os, sys
import pandas as pd
import random
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--target_speaker',type=str, required=True)
parser.add_argument('--output_header',type=str, default='temp')
args=parser.parse_args()

target_speaker = args.target_speaker
# CSVファイルの読み込み
df = pd.read_csv(args.input_csv)
df_train = df.query('speaker!=@target_speaker')
df_valid = df.query('speaker==@target_speaker')

# 出力ファイル名
valid_file = args.output_header + "_valid.csv"
train_file = args.output_header + "_train.csv"

# CSVファイルとして出力
df_train.to_csv(train_file, index=False)
df_valid.to_csv(valid_file, index=False)
