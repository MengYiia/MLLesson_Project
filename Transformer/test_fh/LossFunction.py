import torch
from torch import nn


class MaeAndMseLoss(nn.Module):
    def __init__(self):
        super(MaeAndMseLoss, self).__init__()
        # MAELoss
        self.mae_criterion = torch.nn.L1Loss()
        # MSELoss
        self.mse_criterion = torch.nn.MSELoss()
        # MSELoss的权重
        self.mse_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # 将mse_weight限制在0到1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        mae_loss = self.mae_criterion(pred, target)
        mse_loss = self.mse_criterion(pred, target)
        total_loss = (1 - self.sigmoid(self.mse_weight)) * mae_loss + self.sigmoid(self.mse_weight) * mse_loss
        return total_loss
