import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from Transformer.test_fc.LossFunction import MaeAndMseLoss
from Transformer.test_fc.model import iTransformer
from DataSet import ETTh1
from utils import *


def train(model, dataset_path, batch_size, lr, Epochs, logger, save_model_dir, weight_decay,
          min_val_loss=1e100) -> None:
    """
    训练函数
    :param model: 要训练的模型
    :param dataset_path: 数据集路径
    :param batch_size: 批处理大小
    :param lr: 学习率
    :param Epoch: 总的训练迭代次数
    :param logger: 日志器
    :param save_model_dir: 模型保存路径
    :param weight_decay: 正则化系数
    :return: None
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备{torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path)
    x_train, y_train = dataset.get_data("train")
    x_val, y_val = dataset.get_data("val")

    x_train, y_train, x_val, y_val = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device)

    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss().to(device)

    batch_loss_list = []


    for epoch in range(Epochs):
        model.train()  # 切换回训练模式

        pbar = tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{Epochs}', postfix={'loss': '?'}, mininterval=0.3)
        total_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            X_batch = x_train[start_idx: end_idx]
            Y_batch = y_train[start_idx: end_idx]
            output = model(X_batch)
            # 这里应该不用打平[batch, 7]
            # [batch, 96, 7]才要打平
            # 因为MSE是在最后一个维度计算的
            # 如果是[batch, 96, 7]， 其实是不太符合输入规则的
            # [batch, 7]就是正常的输入shape了，就是7个预测出来的变量和实际值算均方差，最后batch求平均
            # loss = criterion(torch.flatten(output, start_dim=0, end_dim=1),
            #                  torch.flatten(Y_batch, start_dim=0, end_dim=1))
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
            batch_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'loss': total_loss / (i + 1)})
            pbar.update(1)
        pbar.close()
        if (epoch + 1) % 10 != 0:
            continue
        model.eval()
        logger.info('开始验证')
        with torch.no_grad():
            val_loss = 0.0
            num_val_batches = len(x_val) // batch_size
            pbar = tqdm(total=num_val_batches, desc=f'Epoch:{epoch + 1}/{Epochs}', postfix={'val_loss': '?'},
                        mininterval=0.3)
            # 批处理
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_val_batch = x_val[start_idx:end_idx]
                Y_val_batch = y_val[start_idx:end_idx]
                y_pre = model(X_val_batch[:, 0:96, :]).unsqueeze(1)
                for index in range(1, 96):
                    test_x = X_val_batch[:, index:96, :]
                    temp_x = torch.concat((test_x, y_pre), dim=1)
                    tmp_pre = model(temp_x).unsqueeze(1)
                    y_pre = torch.concat((y_pre, tmp_pre), dim=1)
                val_loss += criterion(y_pre, Y_val_batch).item()
                avg_val_loss = val_loss / (i + 1)
                pbar.set_postfix(**{'val_loss': avg_val_loss})
                pbar.update(1)
        pbar.close()
        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            logger.info('保存效果最好的模型:best_epoch_model_newStart_layer3.pth')
            model_dict = model.state_dict()
            model_dict['min_val_loss'] = avg_val_loss
            torch.save(model_dict, os.path.join(save_model_dir, "best_epoch_model_newStart_layer3.pth"))

    # 绘损失图
    plt.plot(batch_loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':

    # 模型参数
    embed_dim = 7
    n_heads = 7
    n_layers = 3
    src_len = 96
    dim_k = 2
    dim_v = 2

    model = iTransformer(embedding_dim=embed_dim,
                         num_heads=n_heads,
                         num_layer=n_layers,
                         result_size=embed_dim,
                         seq_len=src_len,
                         k_dim=dim_k,
                         v_dim=dim_v,
                         ffn_mode='linear')

    save_model_dir = "../model_FcTransformer"
    model_path = "../model_FcTransformer/best_epoch_model_newStart_layer3.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        min_val_loss = 1e100

    logger = setting_logging('iTransformerForETTh1')
    # 定义超参数
    lr = 1e-4
    Epochs = 40
    batch_size = 256
    dataset_path = "../../data/ETTh1.csv"

    weight_decay = 1e-5

    train(model, dataset_path, batch_size, lr, Epochs, logger, save_model_dir, weight_decay, min_val_loss)
