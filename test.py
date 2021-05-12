import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def Data_preprocessing():
    # Load data
    stock = fdr.DataReader('068270', '2000-01-01', '2019-12-31')

    # Convert numeric values between 0~1
    for i in range(len(stock.columns)-1):
        stock[stock.columns[i]] /= stock[stock.columns[i]].max()

    # split year 2005~2018 & 2019
    stock['Year'] = stock.index.year
    stock_train = stock[: len(stock)-len(stock.Year['2019'])]
    stock_test = stock[len(stock)-len(stock.Year['2019']) :]

    stock_train = stock_train.drop('Year', axis=1)
    stock_test = stock_test.drop('Year', axis=1)

    return stock_train, stock_test


def Linear(stock):
    # Seperate data
    y = stock.Close
    x = stock.drop('Close', axis=1)
    x = x.drop('Change', axis=1)
    print(y.dtypes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))

    return model


def Prediction(model, stock_test):
    y = stock_test.Close
    x = stock_test.drop('Close', axis=1)
    x = x.drop('Change', axis=1)

    y_pred = model.predict(x)

    # Visualization
    plt.figure(figsize=(16, 9))
    sns.lineplot(y=y, x=x.index)
    sns.lineplot(y=y_pred, x=x.index)
    plt.legend(['Real-data', 'Predicted-data'])
    plt.show()



if __name__ == '__main__':
    print('start')
    Stock_train, Stock_test = Data_preprocessing()
    Model = Linear(Stock_train)
    Prediction(Model, Stock_test)
