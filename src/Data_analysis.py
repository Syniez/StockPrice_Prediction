import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Stock_data:
    def __init__(self):
        # Load stock price data
        self.stock = fdr.DataReader('068270', '2000-01-01', '2019-12-31')


    def Preprocessing(self):
        self.stock_copy = self.stock.copy()

        # Convert numeric values between 0~1
        for i in range(len(self.stock.columns)-1):
            self.stock[self.stock.columns[i]] /= self.stock[self.stock.columns[i]].max()
       
        # Split year 2005~2018(train) & 2019(test)
        self.stock['Year'] = self.stock.index.year
        self.stock_train = self.stock[: len(self.stock)-len(self.stock.Year['2019'])]
        self.stock_test = self.stock[len(self.stock)-len(self.stock.Year['2019']) :]

        # Remove unused columns
        self.stock_train = self.stock_train.drop('Change', axis=1)
        self.stock_test = self.stock_test.drop('Change', axis=1)
        self.stock_train = self.stock_train.drop('Year', axis=1)
        self.stock_test = self.stock_test.drop('Year', axis=1)
        
        # Split features and labels
        self.stock_train_feature = self.stock_train.drop('Close', axis=1)
        self.stock_train_label = self.stock_train.Close
        self.stock_test_feature = self.stock_test.drop('Close', axis=1)
        self.stock_test_label = self.stock_test.Close

        # Make stock data into sequential data
        self.stock_train_feature = self.Make_sequential(self.stock_train_feature, 20)
        self.stock_test_feature = self.Make_sequential(self.stock_test_feature, 20)
        self.stock_train_label = self.stock_train_label[20:]
        self.stock_test_label = self.stock_test_label[20:]

        return self.stock_train_feature, self.stock_train_label,  self.stock_test_feature, self.stock_test_label, self.stock_copy

    
    def Make_sequential(self, feature, window_size):
        iter = feature.shape[0] - window_size

        sequential_feature = []
        for i in range(iter):
            features = feature[i:i+window_size]
            sequential_feature.append(features)

        return np.array(sequential_feature)
    

    def Analysis(self):
        print("\nStock data visualization\n")
        # print data
        print(self.stock.head(20))

        print("\n")
        # scale
        self.stock_scaled = self.stock.copy()
        for i in range(len(self.stock.columns)-1):
            print(i, "th column's max value : ", self.stock_scaled[self.stock_scaled.columns[i]].max())
            self.stock_scaled[self.stock_scaled.columns[i]] /= self.stock_scaled[self.stock_scaled.columns[i]].max()

        print("\n", self.stock_scaled.head(20))
        
        # entire data plotting
        _, pos = plt.subplots(nrows=2, ncols=3, figsize=(16,9))
        pos[0, 0].plot(self.stock["Open"])
        pos[0, 0].set_title("Open")
        pos[0, 1].plot(self.stock["High"])
        pos[0, 1].set_title("High")
        pos[0, 2].plot(self.stock["Low"])
        pos[0, 2].set_title("Low")
        pos[1, 0].plot(self.stock["Close"])
        pos[1, 0].set_title("Close")
        pos[1, 1].plot(self.stock["Volume"])
        pos[1, 1].set_title("Volume")
        pos[1, 2].plot(self.stock["Change"])
        pos[1, 2].set_title("Change")
        plt.show()

        self.stock['Year'] = self.stock.index.year
        self.trainset = self.stock[: len(self.stock)-len(self.stock.Year['2019'])]
        self.testset = self.stock[len(self.stock)-len(self.stock.Year['2019']) :]

        # target data plotting
        plt.figure(figsize=(16,9))
        plt.plot(self.trainset["Close"], label="Train data")
        plt.plot(self.testset["Close"], label="Test data")
        plt.title("Close data of stock market")
        plt.legend()
        plt.show()
