import torch
from torch import nn


class MaeAndMseLoss(nn.Module):
    def __init__(self):
        super(MaeAndMseLoss, self).__init__()
        # MAELoss
        self.mae_criterion = torch.nn.L1Loss()
        # MSELoss
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, pred, target):
        mae_loss = self.mae_criterion(pred, target)
        mse_loss = self.mse_criterion(pred, target)
        total_loss = 0.5 * mae_loss + 0.5 * mse_loss
        return total_loss
