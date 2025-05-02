import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def corn_loss(logits, labels):
    """
    logits: Tensor of shape (B, K-1)
    labels: Tensor of shape (B,) with values in 1..K
    """
    if logits.ndim != 2:
        logits = logits.unsqueeze(0)  # shape (1, K-1)
    if labels.ndim != 1:
        labels = labels.unsqueeze(0)  # shape (1,)

    batch_size, K_minus_1 = logits.shape
    loss = 0.0

    for k in range(K_minus_1):
        # マスク：このロジットに関係するサンプル
        mask = (labels > k).squeeze()  # shape: (B,)
        if mask.sum() == 0:
            continue

        logit_k = logits[:, k]  # shape: (B,)
        target_k = (labels > (k + 1)).float().squeeze()  # shape: (B,)

        # 保険：logit_k, target_k, mask が0次元にならないように
        if logit_k.ndim == 0:
            logit_k = logit_k.unsqueeze(0)
        if target_k.ndim == 0:
            target_k = target_k.unsqueeze(0)
        if mask.ndim == 0:
            mask = mask.unsqueeze(0)

        # マスクを使って有効なサンプルだけ抽出
        indices = torch.nonzero(mask, as_tuple=True)[0]
        logit_k_masked = logit_k[indices]
        target_k_masked = target_k[indices]

        # バイナリクロスエントロピー（平均）
        loss_k = F.binary_cross_entropy_with_logits(logit_k_masked, target_k_masked, reduction='mean')
        loss += loss_k

    return loss / K_minus_1




