import torch
import torch.nn as nn

class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, weight_type='quadratic'):
        super().__init__()
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
