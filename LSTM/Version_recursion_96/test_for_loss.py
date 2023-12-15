import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from LSTM.Version_recursion_96.DataSet import ETTh1
from LSTM.Version_recursion_96.model import MultivariateLSTM
from LSTM.Version_recursion_96.utils import setting_logging



def test(model, dataset_path, logger, batch_size, Epoches):

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path)
    x_test, y_test = dataset.get_data("test")
    x_test, y_test = x_test.to(device), y_test.to(device)

    mae_list, mse_list = [], []
    criterion_mse = torch.nn.MSELoss().to(device)
    criterion_mae = torch.nn.L1Loss().to(device)

    model.eval()
    for epoch in range(Epoches):
        with torch.no_grad():
            num_val_batches = len(x_test) // batch_size
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_val_batch = x_test[start_idx:end_idx]
                Y_val_batch = y_test[start_idx:end_idx]
                y_pre = model(X_val_batch).unsqueeze(1)
                for index in range(1, 96):
                    test_x = X_val_batch[:, index:96, :]
                    temp_x = torch.concat((test_x, y_pre), dim=1)
                    tmp_pre = model(temp_x).unsqueeze(1)
                    y_pre = torch.concat((y_pre, tmp_pre), dim=1)
                mse_list.append(criterion_mse(y_pre, Y_val_batch).item())
                mae_list.append(criterion_mae(y_pre, Y_val_batch).item())
        print(f"第{epoch + 1}个epoch跑完了哥。")
    print('mse:', np.mean(mse_list))
    print('mse_std:', np.std(mse_list))
    print('mae:', np.mean(mae_list))
    print('mae_std:', np.std(mae_list))

if __name__ == '__main__':
    # 模型参数
    input_size = 7  # 每个时间步的输入特征数
    hidden_size = 32
    output_size = 7  # 每个时间步的输出特征数
    num_layers = 4
    batch_size = 128
    Epoches = 10

    # 实例化模型, 优化器, 损失函数
    model = MultivariateLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers)

    model_path = "../model_recursion/best_epoch_model_96_layer4_2.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        raise ValueError("模型不存在啊啊啊啊啊啊啊啊啊啊！")

    logger = setting_logging('LSTMForETTh1_test')
    dataset_path = "../../data/ETTh1.csv"
    test(model, dataset_path, logger, batch_size, Epoches)