# CORALベース順序回帰モデル（SMIL/MFCC対応・9段階, 重み共有実装）
import torch
import torch.nn as nn
import torch.nn.functional as F


def logits_to_label(logits):
    """
    CORAL の出力（logits）をラベルに変換．
    logits: Tensor of shape (batch_size, num_classes - 1) のシグモイド前の値
    Returns:
        Tensor of shape (batch_size,) with values in [1, num_classes]
    """
    probs = torch.sigmoid(logits)
    # 各閾値を超えた回数 + 1 が予測ラベル
    return (probs > 0.5).sum(dim=1) + 1


class OrdinalRegressionModel(nn.Module):
    def __init__(self,
                 num_mfcc=40,
                 num_smile_feats=88,
                 gru_hidden_dim=64,
                 embed_dim=64,
                 num_classes=9,
                 dropout_rate=0.3,
                 use_mfcc=True,
                 use_smile=True):
        """
        CORAL実装: 重み共有 + 閾値ベクトル

        Args:
            num_mfcc (int): MFCC の次元数 (例: 40)
            num_smile_feats (int): OpenSMILE 特徴量の次元数 (例: 88)
            gru_hidden_dim (int): Bi-GRU の隠れ層次元
            embed_dim (int): SMIL 特徴量埋め込み次元
            num_classes (int): 順序尺度のクラス数 (例: 9)
            dropout_rate (float): ドロップアウト率
            use_mfcc (bool): MFCC を入力に利用するか
            use_smile (bool): SMIL 特徴量を入力に利用するか
        """
        super().__init__()
        self.use_mfcc = use_mfcc
        self.use_smile = use_smile
        self.num_classes = num_classes

        if self.use_mfcc:
            self.gru = nn.LSTM(
                num_mfcc,
                gru_hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.gru_dropout = nn.Dropout(dropout_rate)
        if self.use_smile:
            self.smile_fc = nn.Linear(num_smile_feats, embed_dim)
            self.smile_dropout = nn.Dropout(dropout_rate)

        # 特徴結合後の次元
        input_dim = (
            (gru_hidden_dim * 2) if use_mfcc else 0
        ) + (
            embed_dim if use_smile else 0
        )

        # 共有ユニット
        self.shared_fc = nn.Sequential(
             nn.Linear(input_dim, input_dim // 2),
             nn.ReLU(),
             nn.Dropout(dropout_rate),
             nn.Linear(input_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)  # これは extract_features 用に残してもよい

        # 閾値パラメータ (num_classes - 1)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def extract_features(self, mfcc=None, lengths=None, smile_feats=None):
        """
        入力特徴量を処理して結合ベクトルを返す

        Args:
            mfcc (Tensor): MFCC 系列, shape = (batch_size, seq_len, num_mfcc)
            lengths (Tensor): 各系列の長さ, shape = (batch_size,)
            smile_feats (Tensor): SMIL 特徴量, shape = (batch_size, num_smile_feats)

        Returns:
            Tensor: 結合後特徴量, shape = (batch_size, input_dim)
        """
        parts = []
        if self.use_mfcc:
            packed = nn.utils.rnn.pack_padded_sequence(
                mfcc,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            # 双方向GRUの隠れ状態を結合
            h_fwd, h_bwd = hidden[0], hidden[1]
            out_gru = torch.cat([h_fwd, h_bwd], dim=1)
            parts.append(self.gru_dropout(out_gru))
        if self.use_smile:
            embed = F.relu(self.smile_fc(smile_feats))
            parts.append(self.smile_dropout(embed))
        return torch.cat(parts, dim=1)

    def forward(self, mfcc=None, lengths=None, smile_feats=None):
        """
        ロジット値を計算して返す

        Args:
            mfcc (Tensor): MFCC 系列, shape = (batch_size, seq_len, num_mfcc)
            lengths (Tensor): 各系列の長さ, shape = (batch_size,)
            smile_feats (Tensor): SMIL 特徴量, shape = (batch_size, num_smile_feats)

        Returns:
            Tensor: ロジット, shape = (batch_size, num_classes - 1)
        """
        # 特徴抽出
        x = self.extract_features(mfcc, lengths, smile_feats)
        g = self.shared_fc(x)  # Dropoutは shared_fc 内で処理済み
        logits = g.repeat(1, self.num_classes - 1) - self.thresholds.view(1, -1)
        return logits

    def predict(self, mfcc=None, lengths=None, smile_feats=None):
        """
        ラベル予測まで一貫実行

        Args:
            mfcc (Tensor): MFCC 系列, shape = (batch_size, seq_len, num_mfcc)
            lengths (Tensor): 各系列の長さ, shape = (batch_size,)
            smile_feats (Tensor): SMIL 特徴量, shape = (batch_size, num_smile_feats)

        Returns:
            Tensor: 予測ラベル, shape = (batch_size,) with values in [1, num_classes]
        """
        logits = self.forward(mfcc, lengths, smile_feats)
        return logits_to_label(logits)
