import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


# from pandas import read_csv
# from matplotlib import pyplot
#
# # load dataset
# dataset = read_csv('data/ETTh1.csv', header=0, index_col=0)
# values = dataset.values
# # specify columns to plot
# groups = [0, 1, 2, 3, 4, 5, 6]
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()


def data_load(dataset, n_past):
    dataX, dataY = [], []
    for i in range(n_past, len(dataset) - n_past):
        dataX.append(dataset[i - n_past:i, 0:])
        dataY.append(dataset[i: i + n_past, 0:])
    return np.array(dataX), np.array(dataY)


step = 96

df = pd.read_csv("data/ETTh1.csv", parse_dates=["date"], index_col=[0])
df = df[['OT','HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'HUFL']]

df_for_training = df[:int(0.6 * len(df))]

scaler = MinMaxScaler(feature_range=(-1, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)

x_train, y_train = data_load(df_for_training_scaled, step)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)



x_test = torch.concat((x_train[0], x_train[0 + 96], x_train[96 + 96], x_train[96+96+96]), dim=0)
print(x_test[:336].shape)
print(len(x_train[0][0]))

# x_train[1][-1]  = y_train[0][0]
# print(y_train[:, 0, :].shape)