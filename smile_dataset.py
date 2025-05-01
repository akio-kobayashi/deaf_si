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
import opensmile

# ----------------------------------------------------------------
# 1) eGeMAPSv02 の機能名と6大カテゴリに対応するインデックスを事前計算
# ----------------------------------------------------------------
_smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
feature_names = _smile.feature_names  # List[str], 長さ88

# 6大カテゴリの正規表現パターン
_category_patterns = {
    "F0":       r"^F0_",
    "Energy":   r"loudness|energy",
    "Spectral": r"alphaRatio|hammarbergIndex|spectralSlope|spectralFlux",
    "Jitter":   r"jitter",
    "Shimmer":  r"shimmer",
    "MFCC":     r"mfcc[1-9]$|mfcc1[0-2]|deltaMFCC",
}

# カテゴリ→インデックス一覧 の辞書を作成
category_indices = {}
for cat, pat in _category_patterns.items():
    idxs = [
        i for i, name in enumerate(feature_names)
        if re.search(pat, name, flags=re.IGNORECASE)
    ]
    category_indices[cat] = idxs

def exclude_smile_categories(smile_feats: Tensor, exclude_cats: List[str]) -> Tensor:
    """
    smile_feats: Tensor of shape (..., 88)
    exclude_cats: 除外したいカテゴリ名のリスト
    returns: Tensor of shape (..., 88 - sum(num_features_in_each_category))
    """
    # 除外するインデックスを集合にまとめる
    idxs_to_remove = set()
    for cat in exclude_cats:
        idxs_to_remove.update(category_indices.get(cat, []))
    # 残すインデックス
    idxs_keep = [i for i in range(len(feature_names)) if i not in idxs_to_remove]
    # 最後の次元だけインデックス指定で抽出
    return smile_feats[..., idxs_keep]

# ----------------------------------------------------------------
# 2) SmileDataset クラス（ablation対応）
# ----------------------------------------------------------------
class SmileDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path: str,
        stat_path: str,
        exclude_smile_cats: List[str] = None
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ablation で除きたいカテゴリ名を保持
        self.exclude_smile_cats = exclude_smile_cats or []

        # 事前計算した統計量をロード
        stats = torch.load(stat_path)
        self.smile_mean = stats['smile_mean'].to(self.device)  # (1,88)
        self.smile_var  = stats['smile_var'].to(self.device)   # (1,88)
        self.mfcc_mean  = rearrange(stats['mfcc_mean'].to(self.device), '(b t) -> b t', t=1)
        self.mfcc_var   = rearrange(stats['mfcc_var'].to(self.device), '(b t) -> b t', t=1)

        self.df = pd.read_csv(path)
        self.data_length = len(self.df)

    @staticmethod
    def score_to_rank(score: float) -> int:
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = torch.load(row['feature'])
        # デバイス転送
        smile = data['smile'].to(self.device)  # shape=(1,88)
        mfcc  = data['mfcc'].to(self.device)   # shape=(40,T)

        # 正規化
        mfcc  = (mfcc - self.mfcc_mean) / self.mfcc_var
        smile = (smile - self.smile_mean)     / self.smile_var

        # ablation: 指定カテゴリを除去
        if self.exclude_smile_cats:
            # smile は (1,88) → 最後の次元で除去
            # squeezeして(88,)→ 関数適用後 unsqueeze
            smile_vec = smile.squeeze(0)  # (88,)
            smile_vec = exclude_smile_categories(smile_vec, self.exclude_smile_cats)
            smile = smile_vec.unsqueeze(0)  # (1, 88 - removed)

        rank = self.score_to_rank(row['intelligibility'])
        return mfcc, smile, rank

# ----------------------------------------------------------------
# 3) data_processing 関数は変更なし（smileのshapeは Dataset と合わせてください）
# ----------------------------------------------------------------
def data_processing(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mfccs, smiles, ranks, lengths = [], [], [], []

    for _mfcc, _smile, _rank in data:
        mfccs.append(rearrange(_mfcc, 'f t -> t f'))
        lengths.append(_mfcc.shape[-1])
        smiles.append(_smile)
        ranks.append(_rank)

    ranks   = torch.tensor(ranks, device=device)
    lengths = torch.tensor(lengths, device=device)
    labels  = coral_loss.ordinal_labels(ranks, 9)

    mfccs  = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    smiles = rearrange(nn.utils.rnn.pad_sequence(smiles, batch_first=True),
                       'b c f -> (b c) f')

    return mfccs, smiles, labels, ranks, lengths
