import torch
import torch.nn as nn

class WeightedKappaLoss(nn.Module):

    def weight_matrix(self, num_classes, threshold, device):
        weights = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)
        for i in range(num_classes):
            for j in range(num_classes):
                distance = abs(i - j)
                if distance <= threshold:
                    weights[i, j] = distance ** 2  # 距離が閾値以下なら二乗重み
                else:
                    weights[i, j] = distance       # 距離が閾値を超えるなら線形重み
        return weights/((num_classes - 1)**2)
    
    def __init__(self, num_classes, mode='quadratic', loss_type='kappa', threshold=2, epsilon=1e-10):
        super().__init__()
        self.num_classes = num_classes
        self.y_pow = 2
        if mode == 'linear':
            self.y_pow = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if loss_type == 'kappa':
            repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(device)
            repeat_op_sq = torch.square((repeat_op - repeat_op.T))
            self.weights = repeat_op_sq / ((num_classes - 1) ** 2)
        else:
            self.weights = self.weight_matrix(num_classes, threshold, device)
            
        self.epsilon = epsilon

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).to(y_pred.device)
        y_true = y[y_true]

        y_true = y_true.float()

        #repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(y_pred.device)
        #repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        #weights = repeat_op_sq / ((num_classes - 1) ** 2)

        y_pred = torch.softmax(y_pred, dim=1)
        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(self.weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]),
                                      torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(self.weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)

    import torch
'''
class CustomWeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, threshold=2, epsilon=1.e-10):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.epsilon = epsilon

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    """
    距離に応じて重みを切り替えたWeighted Kappa損失関数．
    
    引数:
    - predictions: モデルの予測テンソル（形状：[batch_size, num_classes]）
    - targets: 正解ラベルのテンソル（形状：[batch_size]）
    - num_classes: 分類クラスの数
    - threshold: 重みを切り替える距離の閾値
    
    戻り値:
    - バッチ全体のWeighted Kappa損失
    """

    def forward(self, y_pred, y_true):
        # 予測確率を取得
        probs = torch.softmax(y_pred, dim=1)
        preds = torch.argmax(probs, dim=1)
    
        # 観測一致行列を作成
        observed_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32, device=y_pred.device)
        for t, p in zip(y_true, preds):
            observed_matrix[t.long(), p.long()] += 1

        # 期待一致行列を作成
        target_hist = torch.histc(y_true.float(), bins=self.num_classes, min=0, max=self.num_classes-1)
        pred_hist = torch.histc(preds.float(), bins=self.num_classes, min=0, max=self.num_classes-1)
        expected_matrix = torch.outer(target_hist, pred_hist) / y_true.size(0)
    
        # 重み行列を作成：距離に応じて一次または二次の重みを適用
        weight_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32, device=y_pred.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                distance = abs(i - j)
                if distance <= self.threshold:
                    weight_matrix[i, j] = distance ** 2  # 距離が閾値以下なら二乗重み
                else:
                    weight_matrix[i, j] = distance       # 距離が閾値を超えるなら線形重み

        # Kappaスコアを計算
        numerator = (weight_matrix * observed_matrix).sum()
        denominator = (weight_matrix * expected_matrix).sum()
        kappa_score = 1 - (numerator / (denominator + self.epsilon))
    
        return kappa_score
'''
