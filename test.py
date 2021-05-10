import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense


def Data_preprocessing():
    # Load data
    stock = fdr.DataReader('068270', '2000-01-01', '2019-12-31')

    # Visualization
    '''plt.figure(figsize=(16,9))
    sns.lineplot(y=stock['Close'], x=stock.index)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()'''

    return stock


def Model():
    # Hyper parameters
    lr = 0.001

    # Build model
    model = Sequential()
    model.add(LSTM(16, activation='tanh'))
    model.add(Dense(16, activatation='relu'))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='Huber', metrics=['mse'])
    model.fit()


if __name__ == '__main__':
    print('System start..')
    samsung()
