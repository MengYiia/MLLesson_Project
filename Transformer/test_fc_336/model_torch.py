import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # 奇数部分
        if d_model > 1:
            if d_model % 2 == 0:
                positional_encoding[:, 1::2] = torch.cos(position * div_term)
            else:
                positional_encoding[:, 1::2] = torch.cos(position * div_term[0:-1])
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        return x + self.positional_encoding[:x.shape[1], :]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim, seq_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerSeqPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers, num_heads, output_dim, dropout=0.1):
        super(TransformerSeqPredictor, self).__init__()

        self.encoder = TransformerEncoder(input_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          seq_len=seq_len,
                                          num_layers=num_layers,
                                          num_heads=num_heads,
                                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.fc(torch.mean(x, dim=1))
        x = self.fc(x[:, -1, :])  # Consider only the last timestep for prediction
        return x

if __name__ == '__main__':
    # 示例用法：
    input_dim = 7  # 输入维度
    hidden_dim = 128  # 隐藏层维度
    num_layers = 3  # Transformer层的数量
    num_heads = 8  # 注意力头的数量
    output_dim = 7  # 输出维度
    seq_len = 96

    model = TransformerSeqPredictor(input_dim, hidden_dim, seq_len, num_layers, num_heads, output_dim)
    x_train = torch.randn(128, 96, 7)
    # y_shape = [batch_size, 7]
    y_train = model(x_train)
    print(y_train.shape)
