import torch
import torch.nn as nn
import torch.nn.functional as F
from smile_model import OrdinalRegressionModel


class AttentionOrdinalRegressionModel(OrdinalRegressionModel):
    """
    CORAL-based ordinal regression with self-attention on MFCC sequences.
    Inherits feature extraction and CORAL logic from OrdinalRegressionModel.
    """
    def __init__(self,
                 num_mfcc=40,
                 num_smile_feats=88,
                 gru_hidden_dim=64,
                 embed_dim=64,
                 num_classes=9,
                 dropout_rate=0.3,
                 use_mfcc=True,
                 use_smile=True,
                 n_heads=4):
        super().__init__(
            num_mfcc=num_mfcc,
            num_smile_feats=num_smile_feats,
            gru_hidden_dim=gru_hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_mfcc=use_mfcc,
            use_smile=use_smile
        )
        # replace GRU with Transformer for MFCC if attention desired
        if use_mfcc:
            # project MFCC features to embed_dim for attention
            self.mfcc_proj = nn.Linear(num_mfcc, embed_dim)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dropout=dropout_rate
            )
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        # adjust shared_fc input dimension if changed
        input_dim = 0
        if use_mfcc:
            input_dim += embed_dim
        if use_smile:
            input_dim += embed_dim
        # override shared_fc
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        # thresholds remain from base class

    def extract_features(self, mfcc=None, lengths=None, smile_feats=None):
        parts = []
        if self.use_mfcc:
            # mfcc: (batch, seq_len, num_mfcc)
            # project
            x = F.relu(self.mfcc_proj(mfcc))  # (batch, seq_len, embed_dim)
            # transformer expects (seq_len, batch, embed_dim)
            x = x.transpose(0, 1)
            x = self.transformer(x)           # (seq_len, batch, embed_dim)
            x = x.transpose(0, 1)             # (batch, seq_len, embed_dim)
            # mean-pool over time
            mfcc_feat = x.mean(dim=1)         # (batch, embed_dim)
            mfcc_feat = self.dropout(mfcc_feat)
            parts.append(mfcc_feat)

        if self.use_smile:
            # use base embedding for smile
            smile_embed = F.relu(self.smile_fc(smile_feats))  # (batch, embed_dim)
            smile_embed = self.smile_dropout(smile_embed)
            parts.append(smile_embed)

        return torch.cat(parts, dim=1)


class CornModel(nn.Module):
    """
    CORN: Conditional Ordinal Regression Network.
    Models P(y>k | y>k-1) with separate binary heads.
    """
    def __init__(self,
                 num_mfcc=40,
                 num_smile_feats=88,
                 gru_hidden_dim=64,
                 embed_dim=64,
                 num_classes=9,
                 dropout_rate=0.3,
                 use_mfcc=True,
                 use_smile=True):
        super().__init__()
        self.use_mfcc = use_mfcc
        self.use_smile = use_smile
        self.num_classes = num_classes

        # same feature extractors as base
        if self.use_mfcc:
            self.gru = nn.LSTM(num_mfcc,
                              gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True)
            self.gru_dropout = nn.Dropout(dropout_rate)
        if self.use_smile:
            self.smile_fc = nn.Linear(num_smile_feats, embed_dim)
            self.smile_dropout = nn.Dropout(dropout_rate)

        input_dim = (gru_hidden_dim*2 if use_mfcc else 0) + (embed_dim if use_smile else 0)
        # conditional binary classifiers for each threshold
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim//2, 1)
            ) for _ in range(num_classes-1)
        ])

    def extract_features(self, mfcc=None, lengths=None, smile_feats=None):
        parts = []
        if self.use_mfcc:
            packed = nn.utils.rnn.pack_padded_sequence(
                mfcc, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            h_fwd, h_bwd = hidden[0], hidden[1]
            gru_out = torch.cat([h_fwd, h_bwd], dim=1)
            parts.append(self.gru_dropout(gru_out))
        if self.use_smile:
            smile_embed = F.relu(self.smile_fc(smile_feats))
            parts.append(self.smile_dropout(smile_embed))
        return torch.cat(parts, dim=1)

    def forward(self, mfcc=None, lengths=None, smile_feats=None):
        x = self.extract_features(mfcc, lengths, smile_feats)
        logits = [clf(x).squeeze(1) for clf in self.classifiers]
        # (batch, K-1)
        return torch.stack(logits, dim=1)

    def predict(self, mfcc=None, lengths=None, smile_feats=None):
        logits = self.forward(mfcc, lengths, smile_feats)
        # sigmoid probabilities > 0.5
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1) + 1  # ranks


class AttentionCornModel(CornModel):
    """
    CORN model with self-attention on MFCC.
    Inherits CornModel but replaces GRU with Transformer.
    """
    def __init__(self, *args, n_heads=4, embed_dim=64, dropout_rate=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        # replace MFCC encoder if use_mfcc
        if self.use_mfcc:
            # project to embed_dim
            self.mfcc_proj = nn.Linear(self.gru.input_size, embed_dim)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dropout=dropout_rate
            )
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
            # adjust input feature dim
            new_input = embed_dim + (embed_dim if self.use_smile else 0)
            # update classifiers first layer
            for clf in self.classifiers:
                # 1層目の線形変換を置き換え
                clf[0] = nn.Linear(new_input, new_input//2)
                # 4層目（最後の線形変換）も置き換え
                clf[3] = nn.Linear(new_input//2, 1)

    def extract_features(self, mfcc=None, lengths=None, smile_feats=None):
        parts = []
        if self.use_mfcc:
            # (B, T, F) -> project
            x = F.relu(self.mfcc_proj(mfcc))
            x = x.transpose(0,1)
            x = self.transformer(x)
            x = x.transpose(0,1)
            mfcc_feat = x.mean(dim=1)
            parts.append(F.dropout(mfcc_feat, p=self.dropout_rate))
        if self.use_smile:
            smile_embed = F.relu(self.smile_fc(smile_feats))
            parts.append(self.smile_dropout(smile_embed))
        return torch.cat(parts, dim=1)
