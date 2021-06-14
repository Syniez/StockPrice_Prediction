from Data_analysis import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def Train_model(train_feature, train_label):
    model = Sequential()
    model.add(LSTM(16, input_shape = (train_feature.shape[1], train_feature.shape[2]), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_feature, train_label, epochs = 50, batch_size=10, validation_split=.2)

    return model


def Prediction(model, test_feature, test_label, original_data):
    test_label = test_label[:, np.newaxis]

    y_pred = model.predict(test_feature)

    label = original_data["Close"]
    label = label[:, np.newaxis]
    scaler = label.max()

    pred = np.zeros((len(label), 1))
    pred[:len(label)-len(y_pred)] = np.nan
    pred[len(label)-len(y_pred):] = y_pred

    vi = label[len(label)-len(y_pred)] * 0.03

    if label[len(label)-len(y_pred)] - pred[len(label)-len(y_pred)]*scaler >= vi:
        diff = label[len(label)-len(y_pred)] - pred[len(label)-len(y_pred)]*scaler - vi
        pred = pred * scaler + diff
        y_pred = y_pred * scaler + diff
    else:
        pred *= scaler
    
    plt.figure(figsize=(16, 9))
    plt.plot(label, label = "Real data")
    plt.plot(pred, label = 'Predicted value')
    plt.legend()
    plt.show()
