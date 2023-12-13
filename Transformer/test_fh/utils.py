# 输入前96小时，预测后96小时
import logging
import os

import numpy as np


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

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)