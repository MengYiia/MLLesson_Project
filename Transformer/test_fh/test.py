from matplotlib import pyplot as plt
from tqdm import tqdm

from model_fh import *
from DataSet import ETTh1
import os
from utils import *


def train(model, dataset_path, batch_size, lr, Epochs, logger, save_model_dir, min_val_loss=1e100) -> None:
    """
    训练函数
    :param model: 要训练的模型
    :param train_dataset: 训练集
    :param val_dataset: 验证集
    :param batch_size: 批处理大小
    :param lr: 学习率
    :param Epoch: 总的训练迭代次数
    :param logger: 日志器
    :param model_path: 模型保存路径
    :return: None
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path)
    x_train, y_train = dataset.get_data("train")

    x_val, y_val = dataset.get_data("val")

    x_train, y_train, x_val, y_val = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device)

    # 计算总共的批次数（向上取整）
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    batch_loss_list = []
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
            output, () = model(X_batch, Y_batch)
            # 计算均方误差
            loss = criterion(output, Y_batch)
            # loss = criterion(output[:, -1, :], Y_batch[:, -1, :])
            total_loss += loss.item()
            batch_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # loss = criterion(output, Y_batch)
            # 将out和labels reshape成2D张量
            # out_reshaped = output.view(-1, 2)  # 5 * 3 = 15, output_dim=2
            # labels_reshaped = Y_batch.view(-1, 2)
            # 更新平均误差值
            pbar.set_postfix(**{'loss': total_loss / (i + 1)})
            pbar.update(1)
        pbar.close()

        # 验证阶段
        model.eval()
        logger.info('开始验证')
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
                output_val, _ = model(X_val_batch)

                out_reshaped = output_val.view(-1, 2)  # 5 * 3 = 15, output_dim=2
                labels_reshaped = Y_val_batch.view(-1, 2)

                val_loss += criterion(out_reshaped, labels_reshaped).item()

                avg_val_loss = val_loss / (i + 1)
                pbar.set_postfix(**{'val_loss': avg_val_loss})
                pbar.update(1)
        pbar.close()
        model.train()  # 切换回训练模式
        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            logger.info('保存效果最好的模型:best_epoch_model_96.pth')
            model_dict = model.state_dict()
            model_dict['min_val_loss'] = avg_val_loss
            torch.save(model_dict, os.path.join(save_model_dir, "best_epoch_model_96.pth"))

    # 绘损失图
    plt.plot(batch_loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':

    # 模型参数
    embed_dim = 7
    n_heads = 8
    n_layers = 6
    src_len = 96
    target_len = 336
    dim_k = 2
    dim_v = 2

    model = Transformer(embedding_dim=embed_dim,
                        num_heads=n_heads,
                        encoder_num_layers=n_layers,
                        decoder_num_layers=n_layers,
                        encoder_seq_len=src_len,
                        decoder_seq_len=target_len,
                        result_size=embed_dim,
                        k_dim=dim_k,
                        v_dim=dim_v)

    save_model_dir = "model"
    model_path = "model/best_epoch_model.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        min_val_loss = 1e100

    logger = setting_logging('TransformerForETTh1')
    # 定义超参数
    lr = 0.00005
    Epochs = 20
    batch_size = 256
    dataset_path = "../../data/ETTh1.csv"

    train(model, dataset_path, batch_size, lr, Epochs, logger, save_model_dir, min_val_loss)
