import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def setting_logging(log_name):
    """
    设置日志
    :param log_name: 日志名
    :return: 可用日志
    """
    # 第一步：创建日志器对象，默认等级为warning
    logger = logging.getLogger(log_name)
    logging.basicConfig(level="INFO")

    # 第二步：创建控制台日志处理器
    console_handler = logging.StreamHandler()

    # 第三步：设置控制台日志的输出级别,需要日志器也设置日志级别为info；----根据两个地方的等级进行对比，取日志器的级别
    console_handler.setLevel(level="WARNING")

    # 第四步：设置控制台日志的输出格式
    console_fmt = "%(name)s--->%(asctime)s--->%(message)s--->%(lineno)d"
    fmt1 = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(fmt=fmt1)

    # 第五步：将控制台日志器，添加进日志器对象中
    logger.addHandler(console_handler)

    return logger
def data_load(dataset, n_past):
    dataX, dataY = [], []
    for i in range(n_past, len(dataset) - n_past):
        dataX.append(dataset[i - n_past:i, 0:])
        dataY.append(dataset[i: i + n_past, 0:])
    return np.array(dataX), np.array(dataY)



class MultiVarLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiVarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 7)  # 7个变量的预测结果

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output



if __name__ == '__main__':

    logger = setting_logging('LSTMForETTh1')
    # 定义超参数
    lr = 0.0005
    Epochs = 10
    batch_size = 128
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # 设定输入维度和隐藏层维度
    input_dim = 7
    hidden_dim = 64

    step = 96

    df = pd.read_csv("../../data/ETTh1.csv", parse_dates=["date"], index_col=[0])

    df_for_training = df[:int(0.6 * len(df))]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    df_for_training_scaled = scaler.fit_transform(df_for_training)

    x_train, y_train = data_load(df_for_training_scaled, step)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test, y_test = data_load(df_for_training_scaled, step)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    # 实例化模型
    model = MultiVarLSTM(input_dim, hidden_dim)

    # save_model_dir = "LSTM/model_new"
    # model_path = "LSTM/model_new/best_epoch_model_96.pth"
    # if os.path.exists(model_path):
    #     saved_state_dict = torch.load(model_path)
    #     min_val_loss = saved_state_dict['min_val_loss']
    #     saved_state_dict.pop('min_val_loss')
    #     model_recursion.load_state_dict(saved_state_dict)
    # else:
    #     min_val_loss = 1e100


    # 定义输入数据
    input_data = x_train.transpose(0, 1)  # 输入数据维度为(时间步长, batch大小, 变量个数)
    target_data = y_train.transpose(0, 1)

    # 计算总共的批次数（向上取整）
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss().to(device)

    loss_list = []
    model.train()
    for epoch in range(Epochs):
        pbar = tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{Epochs}', postfix={'loss': '?',
                                                                                    'acc': '?'}, mininterval=0.3)
        total_loss = 0.0
        # 开始训练
        for i in range(num_batches):
            # 计算起始和结束索引
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            X_batch = input_data[:, start_idx:end_idx]
            Y_batch = target_data[:, start_idx:end_idx]
            # output.shape = [96, batch_size, 7]
            output = model(X_batch)
            # print("output.shape:", output.shape)
            # print("Y_batch.shape:", Y_batch.shape)
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新平均误差值
            pbar.set_postfix(**{'loss': total_loss / (i + 1)})
            pbar.update(1)
        pbar.close()

    # 开始测试
    model.eval()
    with torch.no_grad():
        # y_pre.shape = [96, 1, 7]
        y_pre = model(x_test.transpose(0, 1)[:,1,:].unsqueeze(1))

    y_pre = y_pre.squeeze(1)
    y_test = torch.concat((y_train[0],y_train[1]), dim=0)

    y_pre = scaler.inverse_transform(y_pre)
    y_test = scaler.inverse_transform(y_test)

    index = 0
    start_index = 0
    # 创建横坐标
    x1 = list(range(start_index, start_index + 192))
    x2 = list(range(start_index + 96, start_index + 192))

    plt.plot(x1, y_test[:, index], marker="*", color="k", label='True')
    plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
    plt.legend()
    plt.title("Test Results: 96 Batches")
    plt.show()

