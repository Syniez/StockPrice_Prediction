from Data_analysis import *
from train import *


def main():
   stock = Stock_data()
   #stock.Analysis()                # This function is just for presentation
   train_data, train_label, test_data, test_label, original_data = stock.Preprocessing()
   model = Train_model(train_data, train_label)
   Prediction(model, test_data, test_label, original_data)


if __name__ == '__main__':
    main()
