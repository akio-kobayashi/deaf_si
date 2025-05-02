import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def corn_loss(logits, labels):
    """
    logits: Tensor of shape (B, K-1)
    labels: Tensor of shape (B,) with values in 1..K
    """
    batch_size, K_minus_1 = logits.shape
    loss = 0.0

    for k in range(K_minus_1):
        # 各しきい値における有効サンプルマスク
        mask = (labels > k)  # shape: (B,) bool

        # 有効サンプルがない場合はスキップ
        if mask.sum() == 0:
            continue

        # P(y > k+1 | y > k)
        target_k = (labels > (k + 1)).float()  # shape: (B,)
        logit_k = logits[:, k]  # shape: (B,)

        # 明示的に1次元以上にしておく（安全策）
        if logit_k.ndim == 0:
            logit_k = logit_k.unsqueeze(0)
        if target_k.ndim == 0:
            target_k = target_k.unsqueeze(0)
        if mask.ndim == 0:
            mask = mask.unsqueeze(0)

        # マスクされたサンプルだけを抽出
        logit_k_masked = logit_k[mask]
        target_k_masked = target_k[mask]

        # 損失（平均）
        loss_k = F.binary_cross_entropy_with_logits(logit_k_masked, target_k_masked, reduction='mean')
        loss += loss_k

    return loss / K_minus_1



