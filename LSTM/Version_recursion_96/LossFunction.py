import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomWeightedLoss(nn.Module):
    def __init__(self):
        super(CustomWeightedLoss, self).__init__()

    def forward(self, predictions, targets):
        # 计算每个变量的平方误差
        squared_errors = (predictions - targets) ** 2

        # 对每个变量的误差应用 softmax 得到权重
        weights = F.softmax(squared_errors, dim=1)
        weights = weights.detach()

        # 加权求和得到最终的损失
        weighted_loss = torch.sum(squared_errors * weights) / len(predictions)

        return weighted_loss
