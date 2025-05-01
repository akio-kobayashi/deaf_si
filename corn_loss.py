import torch
import torch.nn.functional as F

def corn_loss(logits, labels):
    """
    logits: Tensor of shape (batch, K-1)
        出力は「条件付きロジット」g_k(x) = logit for P(y>k | y>k-1)
    labels: Tensor of shape (batch,) with values 1..K
    """
    batch, K1 = logits.shape
    loss = 0.0
    for k in range(K1):
        # このタスクに含めるサンプルのマスク
        # P(y>k|y>k-1) を学習するのはラベルが k以上のサンプルのみ
        mask = (labels > k).float()  # y>k-1 のサンプルを選択
        if mask.sum() == 0:
            continue
        # 条件付き正解ラベル
        # たとえば k=0 (P(y>1)) なら labels>1, k=1 (P(y>2|y>1)) なら labels>2...
        target = (labels > (k+1)).float()
        # ロジットは全サンプル分出ているので、マスクをかけて算出
        logit_k = logits[:, k]
        # BCE＋マスク平均
        loss_k = F.binary_cross_entropy_with_logits(logit_k, target, reduction='none')
        loss += (loss_k * mask).sum() / mask.sum()
    return loss / K1
