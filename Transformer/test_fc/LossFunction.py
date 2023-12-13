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


class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, X, Y):
        B, N, M = X.size(0), X.size(1), Y.size(1)

        D = self.compute_distance_matrix(X, Y)
        D.requires_grad_()

        # Initialize the DP matrix
        R = torch.full((B, N + 2, M + 2), float('inf'), dtype=X.dtype, device=X.device)
        R[:, 0, 0] = 0

        # Compute the DP matrix
        for n in range(1, N + 1):
            for m in range(1, M + 1):
                cost = D[:, n - 1, m - 1]
                if self.normalize:
                    cost /= (n + m)
                r0 = R[:, n - 1, m - 1]
                r1 = R[:, n - 1, m]
                r2 = R[:, n, m - 1]
                R[:, n, m] = cost + torch.min(torch.min(r0, r1), r2, out=R[:, n, m])
        # Compute the Soft-DTW value
        return torch.sum(self.softmin(R[:, 1:N + 1, 1:M + 1])) / self.gamma

    def compute_distance_matrix(self, X, Y):
        B, N, M, D = X.size(0), X.size(1), Y.size(1), X.size(2)
        X_sq = torch.sum(X ** 2, dim=2, keepdim=True)
        Y_sq = torch.sum(Y ** 2, dim=2, keepdim=True)
        XY = torch.matmul(X, Y.transpose(1, 2))
        D = X_sq - 2 * XY + Y_sq.transpose(1, 2)
        return D

    def softmin(self, R):
        return -self.gamma * torch.logsumexp(-R / self.gamma, dim=2)
