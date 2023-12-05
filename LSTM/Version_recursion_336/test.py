import os.path
import torch
import matplotlib.pyplot as plt

from LSTM.Version_recursion_96.DataSet import ETTh1
from LSTM.Version_recursion_96.model import MultivariateLSTM
from LSTM.Version_recursion_96.utils import setting_logging


def test_1(model, dataset_path, logger, start_index, index):
    """
    训练脚本1, 滚轮式预测。输入长度96, 输出长度96。
    :param model: 模型
    :param dataset_path: 数据集路径
    :param logger: 日志器
    :param start_index: 测试起始位置
    :param index: 绘图变量
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path, step=96)
    x_test, _ = dataset.get_data("test")
    x_test = x_test.to(device)

    y_pre = []
    temp_test_x = x_test[0 + start_index]
    with torch.no_grad():
        for i in range(336):
            # 0-95 -> 96*, 1-96* -> 97*, 2-96* 97* -> 98* ...
            test_x = temp_test_x[i:96]
            temp_x = torch.concat((test_x, torch.Tensor(y_pre).to(device)), dim=0)
            # 0-95 -> 96*, 0-95、96* -> 97*, 0-95、96*、97* -> 98* ...
            # temp_x = torch.concat((temp_test_x[0: 96], torch.Tensor(y_pre).to(device)), dim=0)
            temp_x = temp_x.unsqueeze(0)
            tmp_pre = model(temp_x)
            y_pre.append(tmp_pre.squeeze(0).tolist())

    y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96], x_test[start_index + 96 * 2], x_test[start_index + 96 * 3], x_test[start_index + 96 * 4]), dim=0)[: 432].tolist()

    y_temp_test = dataset.inverse_fit_transform(y_temp_test)
    y_pre = dataset.inverse_fit_transform(y_pre)


    # 创建横坐标
    x1 = list(range(start_index, start_index + 432))
    x2 = list(range(start_index + 96, start_index + 432))

    plt.plot(x1, y_temp_test[:, index], marker="*", color="k", label='True')
    plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
    plt.legend()
    plt.title("Test Results: 96 Hours -> 336 Hours")
    plt.show()


# def test_2(model_recursion, dataset_path, logger, start_index, index):
#     """
#     训练脚本2, 0-95:96, 1-96:97。输入长度: 96批次 96 * 96条数据 , 输出长度96。
#     :param model_recursion: 模型
#     :param dataset_path: 数据集路径
#     :param logger: 日志器
#     :param start_index: 测试起始位置
#     :param index: 绘图变量
#     """
#     device = ('cuda' if torch.cuda.is_available() else 'cpu')
#     if torch.cuda.is_available() is True:
#         logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
#     model_recursion = model_recursion.to(device)
#
#     dataset = ETTh1(dataset_path)
#     x_test, _ = dataset.get_data("test")
#     x_test = x_test.to(device)
#
#
#     with torch.no_grad():
#         y_pre = model_recursion(x_test)
#
#     y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96]), dim=0).tolist()
#     y_pre = y_pre[start_index + 96: start_index + 192].to('cpu')
#
#     y_temp_test = dataset.inverse_fit_transform(y_temp_test)
#     y_pre = dataset.inverse_fit_transform(y_pre)
#
#     # 创建横坐标
#     x1 = list(range(start_index, start_index + 192))
#     x2 = list(range(start_index + 96, start_index + 192))
#
#     plt.plot(x1, y_temp_test[:, index], marker="*", color="k", label='True')
#     plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
#     plt.legend()
#     plt.title("Test Results: 96 Batches")
#     plt.show()



if __name__ == '__main__':
    # 模型参数
    input_size = 7  # 每个时间步的输入特征数
    hidden_size = 50
    output_size = 7  # 每个时间步的输出特征数
    num_layers = 1
    index = -1
    start_index = 0

    # 实例化模型, 优化器, 损失函数
    model = MultivariateLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers)

    model_path = "../model_recursion/best_epoch_model_336.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        raise ValueError("模型不存在啊啊啊啊啊啊啊啊啊啊！")

    logger = setting_logging('LSTMForETTh1_test')
    dataset_path = "../../data/ETTh1.csv"
    test_1(model, dataset_path, logger, start_index, index)
    # test_2(model_recursion, dataset_path, logger, start_index, index)
