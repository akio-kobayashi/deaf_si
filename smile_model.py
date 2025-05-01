import torch
import torch.nn as nn
import torch.nn.functional as F

def logits_to_label(logits):
    """
    CORAL の出力（logits）をラベルに変換
    logits: (batch, num_classes - 1) のシグモイド前の値
    """
    probs = torch.sigmoid(logits)  # シグモイド関数で確率に変換
    predictions = (probs > 0.5).sum(dim=1) + 1  # 超えたしきい値の数 + 1
    return predictions

class OrdinalRegressionModel(nn.Module):
    def __init__(self, num_mfcc, num_smile_feats, gru_hidden_dim, embed_dim, num_classes, dropout_rate=0.3):
        """
        num_mfcc: MFCCの次元数
        num_smile_feats: OpenSMILEの特徴量の次元数
        gru_hidden_dim: Bi-GRUの隠れ層の次元数
        embed_dim: OpenSMILE特徴量の埋め込み次元
        num_classes: 分類クラス数（例: 1~5なら num_classes=5）
        """
        super(OrdinalRegressionModel, self).__init__()
        
        # Bi-GRU: 可変長のMFCCを固定次元のベクトルに変換
        self.gru = nn.GRU(num_mfcc, gru_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_dropout = nn.Dropout(dropout_rate)
        
        # OpenSMILE特徴量を埋め込みベクトルに変換
        self.smile_fc = nn.Linear(num_smile_feats, embed_dim)
        self.smile_dropout = nn.Dropout(dropout_rate)
        
        # 結合した特徴ベクトルを処理するFeedforward層
        self.fc1 = nn.Linear(gru_hidden_dim * 2 + embed_dim, 128)
        self.fc1_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes - 1)  # CORALでは num_classes - 1

    '''
    def forward(self, mfcc, lengths, smile_feats):
        """
        mfcc: (batch, time, num_mfcc) - 可変長のMFCC特徴
        lengths: (batch, ) - 各バッチの実際の長さ（パディング処理対策）
        smile_feats: (batch, num_smile_feats) - OpenSMILE特徴量
        """
        # --- 1. Bi-GRUによるMFCC処理 ---
        packed = nn.utils.rnn.pack_padded_sequence(mfcc, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (2, batch, hidden_dim)
        hidden_forward = hidden[0, :, :]
        hidden_backward = hidden[1, :, :]
        gru_output = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, gru_hidden_dim * 2)

        # --- 2. OpenSMILE特徴量の埋め込み ---
        smile_embed = F.relu(self.smile_fc(smile_feats))  # (batch, embed_dim)

        # --- 3. 特徴の結合 & Feedforward ---
        combined_features = torch.cat([gru_output, smile_embed], dim=1)  # (batch, gru_hidden_dim*2 + embed_dim)
        x = F.relu(self.fc1(combined_features))
        logits = self.fc2(x)  # (batch, num_classes - 1)
        
        return logits
    '''
    
    def extract_combined_features(self, mfcc, lengths, smile_feats):
        packed = nn.utils.rnn.pack_padded_sequence(mfcc, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (2, batch, hidden_dim)
        hidden_forward = hidden[0, :, :]
        hidden_backward = hidden[1, :, :]
        gru_output = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, gru_hidden_dim * 2)
        gru_output = self.gru_dropout(gru_output)
        
        # --- 2. OpenSMILE特徴量の埋め込み ---
        smile_embed = F.relu(self.smile_fc(smile_feats))  # (batch, embed_dim)
        smile_embed = self.smile_dropout(smile_embed)
        
        # --- 3. 特徴の結合 & Feedforward ---
        combined_features = torch.cat([gru_output, smile_embed], dim=1)  # (batch, gru_hidden_dim*2 + embed_dim)
        return combined_features

    def forward(self, mfcc, lengths, smile_feats):
        combined_features = self.extract_combined_features(mfcc, lengths, smile_feats)
        x = F.relu(self.fc1(combined_features))
        x = self.fc1_dropout(x)
        logits = self.fc2(x)  # (batch, num_classes - 1)
        
        return logits
        
    
    def save_model(self):
        full_path = os.path.join(self.config['logger']['save_dir'],
                                 config['logger']['name'],
                                 'version_' + str(config['logger']['version']),
                                 config['output_path']
                                 )
        torch.save(self.model.to('cpu').state_dict(), full_path)
