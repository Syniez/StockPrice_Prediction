import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from sklearn.model_selection import train_test_split


def Data_preprocessing():
    # Load data
    stock = fdr.DataReader('068270', '2000-01-01', '2019-12-31')

    # Convert numeric values between 0~1
    for i in range(len(stock.columns)-1):
        stock[stock.columns[i]] /= stock[stock.columns[i]].max()
    
    # Visualization
    '''plt.figure(figsize=(16,9))
    sns.lineplot(y=stock['Close'], x=stock.index)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()'''

    return stock


def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


def Model(stock):
    # Seperate data
    y = stock.Close
    x = stock.drop('Close', axis=1)
    x = x.drop('Change', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    train_data = windowed_dataset(y_train, 20, 32, True)
    test_data = windowed_dataset(y_test, 20, 32, False)


    # Hyper parameters
    lr = 0.001

    # Build model
    '''model = Sequential([])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='Huber', metrics=['mse'])
    model.fit()'''


if __name__ == '__main__':
    print('System start..')
    Stock = Data_preprocessing()
    Model(Stock)
