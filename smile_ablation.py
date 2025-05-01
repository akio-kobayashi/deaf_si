import opensmile
import torch
import re

# 1) opensmile インスタンスの生成
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 2) 特徴量名リストを取得 (長さ88)
feature_names = smile.feature_names  # List[str]

# 3) eGeMAPS の６大カテゴリに対応する正規表現パターンを定義
category_patterns = {
    "F0":       r"^F0_",
    "Energy":   r"loudness|energy",
    "Spectral": r"alphaRatio|hammarbergIndex|spectralSlope|spectralFlux",
    "Jitter":   r"jitter",
    "Shimmer":  r"shimmer",
    "MFCC":     r"mfcc[1-9]$|mfcc1[0-2]|deltaMFCC",
}

# 4) 各カテゴリのインデックスリストを作成
category_indices = {}
for cat, pat in category_patterns.items():
    idxs = [
        idx for idx, name in enumerate(feature_names)
        if re.search(pat, name, flags=re.IGNORECASE)
    ]
    category_indices[cat] = idxs

# 5) テンソルからサブテンソルを抽出する関数
def extract_category_feats(smile_feats: torch.Tensor, category: str) -> torch.Tensor:
    """
    smile_feats: Tensor of shape (batch_size, 88)
    category: one of the keys in category_indices
    return: Tensor of shape (batch_size, num_features_in_category)
    """
    idxs = category_indices.get(category)
    if idxs is None:
        raise ValueError(f"Unknown category: {category}")
    # Python のリストで指定すると、自動で新しいテンソルを返します
    return smile_feats[:, idxs]

def extract_excluding_category_feats(smile_feats: torch.Tensor, category: str) -> torch.Tensor:
    """
    カテゴリに該当する特徴を除いたサブテンソルを返す。

    Args:
        smile_feats: Tensor of shape (batch_size, 88)
        category: 除外したいカテゴリ名 (e.g. "F0", "Energy", ...)
    Returns:
        Tensor of shape (batch_size, 88 - num_features_in_category)
    """
    idxs_to_remove = set(category_indices.get(category, []))
    # 残すインデックスは全インデックスから除外リストを差し引く
    idxs_kept = [i for i in range(len(feature_names)) if i not in idxs_to_remove]
    return smile_feats[:, idxs_kept]

if __name__ == '__main__':
  # ---- 使用例 ----
  # (仮に) バッチサイズ16 の SMIL 特徴テンソルがある場合
  batch_size = 16
  smile_feats = torch.randn(batch_size, len(feature_names))  # 16×88 のダミーデータ

  # F0 系だけ取り出す
  f0_feats = extract_category_feats(smile_feats, "F0")  
  print("F0_feats shape:", f0_feats.shape)  # (16, len(category_indices["F0"]))

  # MFCC 系だけ取り出す
  mfcc_feats = extract_category_feats(smile_feats, "MFCC")
  print("MFCC_feats shape:", mfcc_feats.shape)

  # 全カテゴリ分ループで確認
  for cat in category_indices:
      feats = extract_category_feats(smile_feats, cat)
      print(f"{cat}: {feats.shape}")

  # バッチ×88 の SMIL テンソル
  smile_feats = torch.randn(16, 88)

  # F0 系を除いた特徴量を取得
  feats_without_f0 = extract_excluding_category_feats(smile_feats, "F0")
  print(feats_without_f0.shape)  # (16, 88 - len(category_indices["F0"]))

  # Energy 系を除く
  feats_without_energy = extract_excluding_category_feats(smile_feats, "Energy")
  print(feats_without_energy.shape)
