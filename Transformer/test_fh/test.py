import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from Transformer.test_fh.LossFunction import MaeAndMseLoss
from model_fh import *
from DataSet import ETTh1
import os
from utils import *

def test(model, dataset_path, logger, start_index, save_path) -> None:
    """
    测试函数
    :param model: 要测试的模型
    :param dataset_path: 测试集
    :param logger: 日志器
    :param model_path: 模型保存路径
    :return: None
    """

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path)
    x_test, _ = dataset.get_data("test")
    x_test = x_test.to(device)

    model.eval()
    temp_test_x = x_test[0 + start_index].unsqueeze(0)
    with torch.no_grad():
        # 创建decoder第一个输入，全1作为开始符 [batch_size, seq_len, embedding_dim]
        # decoder_input = torch.ones((1, 1, x_test.shape[-1]), requires_grad=False).to(device)

        # decoder输入encoder_input的最后一个值
        decoder_input = temp_test_x[:, -1, :].unsqueeze(1)

        # decoder输入95, encoder_input:0-94
        # decoder_input = temp_test_x[:, -1, :].unsqueeze(1)
        # temp_test_x = temp_test_x[:, :-1, :]

        # 创建decoder第一个输入，全-1作为开始符 [batch_size, seq_len, embedding_dim]
        # decoder_input = (torch.ones((1, 1, x_test.shape[-1]), requires_grad=False) * -1).to(device)

        y_pre, _ = model(temp_test_x, decoder_input)

    y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96]), dim=0).tolist()

    y_temp_test = dataset.inverse_fit_transform(y_temp_test)  # [192, 7]
    y_pre = dataset.inverse_fit_transform(y_pre.squeeze(0).to('cpu').tolist()) # [96,7]

    # y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96]), dim=0).to('cpu')
    # y_pre = y_pre.squeeze(0).to('cpu')

    output_folder = os.path.join("..\..\pic_transformer", save_path)
    create_directory(output_folder)

    for index in range(len(x_test[0][0])):
        # 创建横坐标
        x1 = list(range(start_index, start_index + 192))
        x2 = list(range(start_index + 96, start_index + 192))

        plt.plot(x1, y_temp_test[:, index], marker="*", color="k", label='True')
        plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
        plt.legend()
        plt.title("The %d variable predicted by the transformer" %(index+1))
        # 图片保存路径
        image_path = os.path.join(output_folder, f"variable_{index + 1}_prediction.png")
        plt.savefig(image_path)
        plt.cla()
    plt.close('all')
    print("小姐,图老奴给你画好了。")


if __name__ == '__main__':

    # 模型参数
    embed_dim = 7
    n_heads = 8
    n_layers = 3
    src_len = 96
    target_len = 96
    dim_k = 2
    dim_v = 2

    model = Transformer(embedding_dim=embed_dim,
                        num_heads=n_heads,
                        encoder_num_layers=n_layers,
                        decoder_num_layers=n_layers,
                        encoder_seq_len=src_len,
                        decoder_seq_len=target_len + 1,  # 由于考虑到开始符, decoder序列长度实际要加一
                        result_size=embed_dim,
                        k_dim=dim_k,
                        v_dim=dim_v)

    save_model_dir = "../model_EnDeTransformer"
    model_path = "../model_EnDeTransformer/best_epoch_model_newStart.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        min_val_loss = 1e100

    logger = setting_logging('TransformerForETTh1')
    # 定义超参数
    lr = 1e-6
    Epochs = 20
    batch_size = 256
    dataset_path = "../../data/ETTh1.csv"

    start_index = 100

    test(model, dataset_path, logger, start_index, 'best_epoch_model_newStart')