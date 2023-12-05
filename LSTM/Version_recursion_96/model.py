import torch
from torch import nn


class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultivariateLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    # 定义 LSTM 模型
    input_size = 10
    hidden_size = 20
    num_layers = 2
    batch_size = 3
    seq_len = 4
    num_directions = 1  # 单向

    lstm = MultivariateLSTM(input_size, hidden_size, 1, num_layers)

    # 创建输入张量
    input_data = torch.randn(batch_size, seq_len, input_size)

    # 前向传播
    output = lstm(input_data)

    # 输出形状
    print("Output shape:", output.shape)
