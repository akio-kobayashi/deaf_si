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
        mask = (labels > k)
        if mask.sum() == 0:
            continue

        target_k = (labels > (k + 1)).float()
        logit_k = logits[:, k]

        # 次元保証（1次元でなければインデクスエラーになるので）
        if logit_k.ndim == 0:
            logit_k = logit_k.unsqueeze(0)
        if target_k.ndim == 0:
            target_k = target_k.unsqueeze(0)
        if mask.ndim == 0:
            mask = mask.unsqueeze(0)

        logit_k = logit_k[mask]
        target_k = target_k[mask]

        loss_k = F.binary_cross_entropy_with_logits(logit_k, target_k, reduction='mean')
        loss += loss_k

    return loss / K1


