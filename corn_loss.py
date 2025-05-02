import torch
import torch.nn.functional as F

def corn_loss(logits, labels):
    """
    logits: Tensor of shape (batch, K-1)
    labels: Tensor of shape (batch,) with values 1..K
    """
    batch, K1 = logits.shape
    loss = 0.0
    for k in range(K1):
        # このタスクに含めるサンプルのマスク（float型）
        mask = (labels > k).float()  # shape: [B]
        if mask.sum() == 0:
            continue
        # 条件付き正解ラベル（float型）
        target_k = (labels > (k + 1)).float()  # shape: [B]
        logit_k = logits[:, k]  # shape: [B]
        
        # 有効サンプルのみ取り出して損失を計算
        logit_k = logit_k[mask.bool()]
        target_k = target_k[mask.bool()]
        
        loss_k = F.binary_cross_entropy_with_logits(logit_k, target_k, reduction='mean')
        loss += loss_k

    return loss / K1

