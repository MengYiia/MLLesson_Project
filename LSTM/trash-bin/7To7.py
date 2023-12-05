import logging
import os.path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm

from LSTM.Version_recursion_96.model import MultivariateLSTM


# 输入前96小时，预测后96小时
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
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:])
        dataY.append(dataset[i, 0:])
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':

    logger = setting_logging('LSTMForETTh1')
    # 定义超参数
    lr = 0.0005
    Epochs = 10
    batch_size = 128
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    step = 96
    df = pd.read_csv("../../data/ETTh1.csv", parse_dates=["date"], index_col=[0])

    df_for_training = df[:int(0.6 * len(df))]
    df_for_valing = df[int(0.6 * len(df)):int(0.8 * len(df))]
    df_for_testing = df[int(0.8 * len(df)):]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_valing_scaled = scaler.fit_transform(df_for_valing)
    df_for_testing_scaled = scaler.transform(df_for_testing)

    x_train, y_train = data_load(df_for_training_scaled, step)
    x_val, y_val = data_load(df_for_valing_scaled, step)
    x_test, y_test = data_load(df_for_testing_scaled, step)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 计算总共的批次数（向上取整）
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    # 模型参数
    input_size = 7  # 每个时间步的输入特征数
    hidden_size = 50
    output_size = 7  # 每个时间步的输出特征数
    num_layers = 1

    # 实例化模型, 优化器, 损失函数
    model = MultivariateLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers)

    save_model_dir = "../model_recursion"
    model_path = "../model_recursion/best_epoch_model_96.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        min_val_loss = 1e100
    model = model.to(device)

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
            X_batch = x_train[start_idx:end_idx]
            Y_batch = y_train[start_idx:end_idx]
            output = model(X_batch)
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
        model.eval()
        logger.info('开始验证')
        # 验证阶段

        with torch.no_grad():
            val_loss = 0.0
            num_val_batches = len(x_val) // batch_size
            pbar = tqdm(total=num_val_batches, desc=f'Epoch:{epoch + 1}/{Epochs}', postfix={'val_loss': '?',
                                                                                            'val_acc': '?'},
                        mininterval=0.3)
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_val_batch = x_val[start_idx:end_idx]
                Y_val_batch = y_val[start_idx:end_idx]
                output_val = model(X_val_batch)
                val_loss += criterion(output_val, Y_val_batch).item()

                avg_val_loss = val_loss / (i + 1)
                pbar.set_postfix(**{'val_loss': avg_val_loss})
                pbar.update(1)
        pbar.close()
        model.train()  # 切换回训练模式
        if avg_val_loss <= min_val_loss:
            logger.info('保存效果最好的模型:best_epoch_model_96.pth')
            model_dict = model.state_dict()
            model_dict['min_val_loss'] = avg_val_loss
            torch.save(model_dict, os.path.join(save_model_dir, "best_epoch_model_96.pth"))
    model.eval()
    # list[tensor]
    y_pre = []
    start_i = 100
    temp_test_x = x_test[0 + start_i]
    with torch.no_grad():
        for i in range(96):
            test_x = temp_test_x[i:96]
            temp_x = torch.concat((test_x, torch.Tensor(y_pre).to(device)), dim=0)
            temp_x = temp_x.unsqueeze(0)
            # [1, 7]
            tmp_pre = model(temp_x)
            y_pre.append(tmp_pre.squeeze(0).tolist())

    y_temp_test = torch.concat((x_test[start_i], x_test[start_i + 96]), dim=0).tolist()

    y_temp_test = scaler.inverse_transform(y_temp_test)
    y_pre = scaler.inverse_transform(y_pre)


    #
    # # # 绘损失图
    # # g1 = plt.figure()
    # # plt.plot(loss_list)
    # # plt.title("Training Loss")
    # # plt.xlabel("Epoch")
    # # plt.ylabel("Loss")
    # #
    # # 绘对比图
    # g2 = plt.figure()
    #
    # 创建横坐标
    x1 = list(range(192))
    x2 = list(range(96, 192))

    plt.plot(x1, y_temp_test[:, 0], marker="*", color="k", label='True')
    plt.plot(x2, y_pre[:, 0], marker="o", color="r", label='Pred')
    plt.legend()
    plt.title("Test Results")
    plt.show()
    #
    # # 模型评估
    # test_rmse = np.sqrt(np.mean(np.square(y_pre, y_test)))
    # test_mae = np.mean(np.abs(y_pre, y_test))
    # print("test_rmse:", test_rmse, "\n", "test_mae:", test_mae)
