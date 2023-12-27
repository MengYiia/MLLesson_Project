import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

from utils import data_load_val_test, data_load_train


class ETTh1:
    def __init__(self, path, step: int = 96, is_scaler: bool = True):
        """
        构造方法
        :param path: 数据集加载路径
        :param step: 拆分数据的滑动窗口大小
        :param is_scaler: 是否进行归一化
        """
        # 拆分数据的滑动窗口大小
        self.step = step

        # 读取原始数据
        self.df_raw = pd.read_csv(path, parse_dates=["date"], index_col=[0])
        # self.df_raw = self.df_raw[['OT', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'HUFL']]
        # self.df_raw = self.df_raw.values
        # 拆分数据集
        self.df = {'train': self.df_raw[:int(0.6 * len(self.df_raw))],
                   'val': self.df_raw[int(0.6 * len(self.df_raw)):int(0.8 * len(self.df_raw))],
                   'test': self.df_raw[int(0.8 * len(self.df_raw)):]}

        self.is_scaler = is_scaler
        if self.is_scaler:
            # 数据归一化工具
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            # 对数据进行归一化
            self.df['train'] = self.scaler.fit_transform(self.df['train'])
            self.df['val'] = self.scaler.fit_transform(self.df['val'])
            self.df['test'] = self.scaler.fit_transform(self.df['test'])

    def get_data(self, mode):
        """
        获取指定数据集
        :param mode: 读取模式
        :return: 拆分好的x, y
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError("请输入正确的读取模式！['train', 'val', 'test']")
        if mode in ['test']:
            x, y = data_load_val_test(self.df[mode], self.step)
        else:
            x, y = data_load_train(self.df[mode], self.step)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def inverse_fit_transform(self, x):
        if self.is_scaler:
            return self.scaler.inverse_transform(x)
        else:
            raise ValueError("未进行归一化操作！")
