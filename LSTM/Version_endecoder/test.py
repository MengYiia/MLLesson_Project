import os.path
import torch
import matplotlib.pyplot as plt

from LSTM.Version_endecoder.DataSet import ETTh1
from LSTM.Version_endecoder.model_Encoder_Decoder import MultivariateLSTM
from LSTM.Version_endecoder.utils import setting_logging


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

    dataset = ETTh1(dataset_path)
    x_test, _ = dataset.get_data("test")
    x_test = x_test.to(device)

    temp_test_x = x_test[0 + start_index].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        y_pre = model(temp_test_x).squeeze(0)
    y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96]), dim=0).tolist()

    y_temp_test = dataset.inverse_fit_transform(y_temp_test)
    y_pre = dataset.inverse_fit_transform(y_pre.to('cpu'))

    # 创建横坐标
    x1 = list(range(start_index, start_index + 192))
    x2 = list(range(start_index + 96, start_index + 192))

    plt.plot(x1, y_temp_test[:, index], marker="*", color="k", label='True')
    plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
    plt.legend()
    plt.title("Test Results: 96 Hours")
    plt.show()



if __name__ == '__main__':
    # 模型参数
    input_size = 7  # 每个时间步的输入特征数
    hidden_size = 50
    output_size = 7  # 每个时间步的输出特征数
    num_layers = 1
    index = 2
    start_index = 100

    # 实例化模型, 优化器, 损失函数
    model = MultivariateLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers, time_size=96)

    model_path = "../model_endecoder/best_epoch_model.pth"
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
