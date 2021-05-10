import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def samsung():
    # Load data
    stock = fdr.DataReader('068270', '2000-01-01', '2019-12-31')
    print(stock.head())

    plt.figure(figsize=(16,9))
    sns.lineplot(y=stock['Close'], x=stock.index)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()


if __name__ == '__main__':
    print('System start..')
    samsung()