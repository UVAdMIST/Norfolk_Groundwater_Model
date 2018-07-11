"""
This network uses the last 26 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-043
"""

import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
from keras.layers import Activation
from keras.utils import plot_model, np_utils
from keras.regularizers import L1L2
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random as rn
import os
matplotlib.rcParams.update({'font.size': 8})


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def data():
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    n_lags = 26
    n_ahead = 19
    n_train = 49563

    dataset_raw = read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_043_no_blanks.csv",
                           index_col=None, parse_dates=True, infer_datetime_format=True)

    train_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[:n_train]
    test_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[n_train:]
    test_dates = test_dates.reset_index(drop=True)
    test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

    dataset = dataset_raw.drop(dataset_raw.columns[[0, 3, 4, 5, 6]], axis=1)

    values = dataset.values
    values = values.astype('float32')

    gwl = values[:, 0]
    gwl = gwl.reshape(gwl.shape[0], 1)

    tide = values[:, 1]
    tide = tide.reshape(tide.shape[0], 1)

    rain = values[:, 2]
    rain = rain.reshape(rain.shape[0], 1)

    gwl_scaler, tide_scaler, rain_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    gwl_scaled = gwl_scaler.fit_transform(gwl)
    tide_scaled = tide_scaler.fit_transform(tide)
    rain_scaled = rain_scaler.fit_transform(rain)

    gwl_super = series_to_supervised(gwl_scaled, n_lags, n_ahead)
    gwl_super_values = gwl_super.values
    tide_super = series_to_supervised(tide_scaled, n_lags, n_ahead)
    tide_super_values = tide_super.values
    rain_super = series_to_supervised(rain_scaled, n_lags, n_ahead)
    rain_super_values = rain_super.values

    gwl_input, gwl_labels = gwl_super_values[:, 0:n_lags + 1], gwl_super_values[:, n_lags + 1:]

    train_X = np.concatenate((gwl_input[:n_train, :], tide_super_values[:n_train, :], rain_super_values[:n_train, :]),
                             axis=1)
    test_X = np.concatenate((gwl_input[n_train:, :], tide_super_values[n_train:, :], rain_super_values[n_train:, :]),
                            axis=1)
    train_y, test_y = gwl_labels[:n_train, :], gwl_labels[n_train:, :]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y, gwl_scaler, test_dates


def create_model(train_X, train_y, test_X, test_y):
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    n_ahead = 19
    n_test = 7548
    n_epochs = 10000
    n_neurons = 10
    n_batch = 49563

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    model = Sequential()
    model.add(LSTM(units={{choice([10, 15, 20, 40, 50, 75])}}, activation={{choice(['relu', 'tanh', 'sigmoid'])}},
                   input_shape=(None, train_X.shape[2]), use_bias=True,
                   bias_regularizer=L1L2(l1=0.01, l2=0.01)))
    model.add(Dropout({{uniform(0.1, 0.5)}}))
    model.add(Dense(activation='linear', units=n_ahead-1, use_bias=True))

    adam = keras.optimizers.Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd = keras.optimizers.SGD(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})

    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model.compile(loss=rmse, optimizer=optim)

    earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=5, verbose=1, mode='auto')
    model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False, callbacks=[earlystop])
    loss = model.evaluate(test_X, test_y, batch_size=n_test, verbose=0)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    n_ahead = 19
    n_lags = 26

    path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_results_18hr/"

    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())
    train_X, train_y, test_X, test_y, gwl_scaler, test_dates = data()
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
