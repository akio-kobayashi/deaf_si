#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
from mord import LogisticAT, OrdinalRidge
from torch.utils.data import DataLoader
from smile_dataset import SmileDataset
import smile_dataset as S

def build_xy(loader, use_mfcc=True, use_smile=True):
    X_parts, y_list = [], []
    for mfcc, smile, _, ranks, lengths in loader:
        parts = []
        if use_mfcc:
            mfcc_avg = mfcc.mean(dim=1)            # (B, D_mfcc)
            parts.append(mfcc_avg.cpu().numpy())
        if use_smile:
            smile_vec = smile.view(-1, smile.shape[-1])
            parts.append(smile_vec.cpu().numpy())
        if not parts:
            raise RuntimeError("Both use_mfcc and use_smile are False!")
        X_parts.append(np.concatenate(parts, axis=1))
        y_list.append(ranks.cpu().numpy())
    return np.vstack(X_parts), np.concatenate(y_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  required=True)
    parser.add_argument("--valid_csv",  required=True)
    parser.add_argument("--stat_path",  required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_mfcc",   action="store_true")
    parser.add_argument("--use_smile",  action="store_true")
    parser.add_argument("--model_type",
                        choices=["LogisticAT", "OrdinalRidge"],
                        default="LogisticAT",
                        help="Which mord model to use")
    parser.add_argument("--alpha",      type=float, default=1.0,
                        help="Regularization strength")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # 1) データセット & ローダー
    ds_train = SmileDataset(path=args.train_csv, stat_path=args.stat_path)
    ds_valid = SmileDataset(path=args.valid_csv, stat_path=args.stat_path)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              collate_fn=S.data_processing)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size,
                              collate_fn=S.data_processing)

    # 2) 特徴行列とラベルを作成
    X_train, y_train = build_xy(train_loader,
                                use_mfcc=args.use_mfcc,
                                use_smile=args.use_smile)
    X_valid, y_valid = build_xy(valid_loader,
                                use_mfcc=args.use_mfcc,
                                use_smile=args.use_smile)

    # 3) モデル初期化 & 学習
    if args.model_type == "LogisticAT":
        clf = LogisticAT(alpha=args.alpha)
    else:  # "OrdinalRidge"
        clf = OrdinalRidge(alpha=args.alpha)
    clf.fit(X_train, y_train)

    # 4) 精度を表示
    train_acc = (clf.predict(X_train) == y_train).mean()
    valid_acc = (clf.predict(X_valid) == y_valid).mean()
    print(f"{args.model_type} Train Acc: {train_acc:.4f}")
    print(f"{args.model_type} Valid Acc: {valid_acc:.4f}")

    # 5) 予測をスコア (1.0,1.5,…,5.0) に戻してCSV出力
    pred_ranks  = clf.predict(X_valid)
    pred_scores = 1.0 + (pred_ranks - 1) * 0.5

    df = pd.read_csv(args.valid_csv)
    df['predict'] = pred_scores
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

#Usage:
#python3 clm_run.py \
#  --train_csv smile/BF026_train.csv \
#  --valid_csv smile/BF026_valid.csv \
#  --stat_path ./stats.pt \
#  --batch_size 16 \
#  --use_mfcc \
#  --use_smile \
#  --model_type OrdinalRidge \
#  --alpha 0.5 \
#  --output_csv BF026_valid_pred.csv

if __name__ == "__main__":
    main()
