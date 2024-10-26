import torch
import torch.nn as nn

class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, mode='quadratic', name='cohen_kappa_loss',epsilon=1e-10):
        super().__init__()
        self.num_classes = num_classes
        if mode == 'quadratic':
            self.y_pow = 2
        if mode == 'linear':
            self.y_pow = 1
        self.epsilon = epsilon

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).to(y_pred.device)
        y_true = y[y_true]

        y_true = y_true.float()

        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(y_pred.device)
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        y_pred = torch.softmax(y_pred, dim=1)
        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]),
                                      torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)
    
