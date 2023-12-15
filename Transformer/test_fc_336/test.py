import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from Transformer.test_fc.LossFunction import MaeAndMseLoss
from Transformer.test_fc.model import iTransformer
from DataSet import ETTh1
from Transformer.test_fc.model_torch import TransformerSeqPredictor
from utils import *

def test_1(model, dataset_path, logger, start_index, save_path):
    """
    滚轮式预测。输入长度96, 输出长度96。
    :param model: 模型
    :param dataset_path: 数据集路径
    :param logger: 日志器
    :param start_index: 测试起始位置
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')
    model = model.to(device)

    dataset = ETTh1(dataset_path)
    x_test, _ = dataset.get_data("test")
    temp_x_test = x_test.to(device)[0 + start_index].unsqueeze(0)

    y_pre = model(temp_x_test).unsqueeze(1)
    with torch.no_grad():
        for index in range(1, 336):
            if index < 96:
                test_x = temp_x_test[:, index:96, :]
                temp_x = torch.concat((test_x, y_pre), dim=1)
            else:
                temp_x = y_pre[:, index-96:, :]
            tmp_pre = model(temp_x).unsqueeze(1)
            y_pre = torch.concat((y_pre, tmp_pre), dim=1)

    y_temp_test = torch.concat((x_test[start_index], x_test[start_index + 96], x_test[start_index + 96 * 2], x_test[start_index + 96 * 3], x_test[start_index + 96 * 4]), dim=0)[: 432].tolist()

    y_temp_test = dataset.inverse_fit_transform(y_temp_test)
    y_pre = dataset.inverse_fit_transform(y_pre.squeeze(0).tolist())


    output_folder = os.path.join("..\..\pic_itransformer_336", save_path)
    create_directory(output_folder)

    for index in range(len(x_test[0][0])):
        # 创建横坐标
        x1 = list(range(start_index, start_index + 432))
        x2 = list(range(start_index + 96, start_index + 432))

        plt.plot(x1, y_temp_test[:, index], marker="*", color="k", label='True')
        plt.plot(x2, y_pre[:, index], marker="o", color="r", label='Pred')
        plt.legend()
        plt.title("The %d variable predicted by the iTransformer" %(index+1))
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
    dim_k = 2
    dim_v = 2
    start_index = 0

    # model = iTransformer(embedding_dim=embed_dim,
    #                     num_heads=n_heads,
    #                     num_layer=n_layers,
    #                     result_size=embed_dim,
    #                     seq_len= src_len,
    #                     k_dim=dim_k,
    #                     v_dim=dim_v,
    #                     ffn_mode='linear')

    model = TransformerSeqPredictor(input_dim=embed_dim, hidden_dim=64, seq_len=src_len, num_heads=n_heads,
                                    num_layers=n_layers, output_dim=embed_dim)

    model_path = "../model_FcTransformer_336/best_epoch_model_layer3_hideen64_torchmodel.pth"
    if os.path.exists(model_path):
        saved_state_dict = torch.load(model_path)
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        raise ValueError("模型不存在啊啊啊啊啊啊啊啊啊啊！")

    logger = setting_logging('iTransformerForETTh1')

    dataset_path = "../../data/ETTh1.csv"

    test_1(model, dataset_path, logger, start_index, 'best_epoch_model_layer3_hideen64_torchmodel')
