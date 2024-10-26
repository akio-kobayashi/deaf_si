import torch
import torch.nn as nn

class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, weight_type='quadratic'):
        super().__init__(self)
        self.num_classes = num_classes
        self.weight_matrix = torch.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if weight_type == 'linear':
                    self.weight_matrix[i, j] = abs(i - j)
                elif weight_type == 'quadratic':
                    self.weight_matrix[i, j] = (i - j) ** 2


    def forward(self, y_pred, y_true):
        # y_trueをone-hotエンコーディング
        y_true_one_hot = nn.functional.one_hot(y_true, num_classes=self.num_classes).float().to(y_pred.device)
        y_pred = torch.softmax(y_pred, dim=1)

        # 観測された混同行列
        observed_matrix = torch.matmul(y_true_one_hot.t(), y_pred)
        # y_trueとy_predの辺りに基づく期待混同行列
        row_sum = y_true_one_hot.sum(dim=0, keepdim=True)
        col_sum = y_pred.sum(dim=0, keepdim=True)
        expected_matrix = torch.matmul(row_sum.t(), col_sum) / y_pred.size(0)

        # 重み付きの観測された一致度
        weighted_observed = torch.sum(self.weight_matrix.to(y_pred.device) * observed_matrix)

        # 重み付きの期待一致度
        weighted_expected = torch.sum(self.weight_matrix.to(y_pred.device) * expected_matrix)

        # Weighted Kappa損失の計算
        kappa_score = 1 - (weighted_observed / weighted_expected)
        return kappa_score
    
def weighted_kappa_loss(y_pred, y_true, num_classes, weight_type='quadratic'):
    """
    y_pred: モデルの予測（確率分布、[batch_size, num_classes]）
    y_true: 実際のラベル（整数ラベル、[batch_size]）
    num_classes: クラスの総数
    weight_type: 重みのタイプ（'linear'または'quadratic'）
    """
    # y_trueをone-hotエンコーディング
    y_true_one_hot = nn.functional.one_hot(y_true, num_classes=num_classes).float()

    # 重み行列の計算
    weight_matrix = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if weight_type == 'linear':
                weight_matrix[i, j] = abs(i - j)
            elif weight_type == 'quadratic':
                weight_matrix[i, j] = (i - j) ** 2

    # y_predの正規化（ソフトマックス）
    y_pred = torch.softmax(y_pred, dim=1)

    # 観測された混同行列
    observed_matrix = torch.matmul(y_true_one_hot.t(), y_pred)

    # y_trueとy_predの辺りに基づく期待混同行列
    row_sum = y_true_one_hot.sum(dim=0, keepdim=True)
    col_sum = y_pred.sum(dim=0, keepdim=True)
    expected_matrix = torch.matmul(row_sum.t(), col_sum) / y_pred.size(0)

    # 重み付きの観測された一致度
    weighted_observed = torch.sum(weight_matrix * observed_matrix)

    # 重み付きの期待一致度
    weighted_expected = torch.sum(weight_matrix * expected_matrix)

    # Weighted Kappa損失の計算
    kappa_score = 1 - (weighted_observed / weighted_expected)
    return kappa_score

# ダミーの予測とラベル
y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.3, 0.4, 0.3]])
y_true = torch.tensor([2, 0, 1])

# Weighted Kappa損失の計算
loss = weighted_kappa_loss(y_pred, y_true, num_classes=3, weight_type='quadratic')

print(f"Weighted Kappa Loss: {loss.item()}")
