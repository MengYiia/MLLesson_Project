import torch
from torch import nn


class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, time_size):
        super(MultivariateLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.time_size = time_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # self.lstm_decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size,
        #                             num_layers=num_layers, batch_first=True)
        # self.fc_decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, y=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)

        _, (h, c) = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :]).unsqueeze(1)

        decoder_start_input = torch.zeros(x.shape[0], 1, self.input_size, dtype=torch.float32).to(x.device)
        out, (h, c) = self.lstm(decoder_start_input, (h, c))
        out = self.fc(out)
        # 第一个输出
        pre_result = out

        if self.training:
            assert y is not None, "模型训练时，y不能为空！"
            for i in range(self.time_size - 1):
                out, (h, c) = self.lstm(y[:, i, :].unsqueeze(1), (h, c))
                out = self.fc(out)
                pre_result = torch.cat((pre_result, out), dim=1)
            return pre_result
        else:
            for i in range(self.time_size - 1):
                out, (h, c) = self.lstm(out, (h, c))
                out = self.fc(out)
                pre_result = torch.cat((pre_result, out), dim=1)
            return pre_result



if __name__ == '__main__':

    # 模型参数
    input_size = 7  # 每个时间步的输入特征数
    hidden_size = 50
    output_size = 7  # 每个时间步的输出特征数
    num_layers = 1
    batch_size = 3
    seq_len = 96

    lstm = MultivariateLSTM(input_size, hidden_size, output_size, num_layers, 96)
    lstm.eval()
    # 创建输入张量
    input_data = torch.randn(batch_size, seq_len, output_size)

    # 前向传播
    output = lstm(input_data)

    # 输出形状
    # 预计输出.shape = [batch_size, time_size, output_size]
    print("Output shape:", output.shape)