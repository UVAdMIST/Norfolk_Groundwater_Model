"""
Model: RNN
Data: storm data set, bootstrapped
Run from shell script
This network uses the last 48 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-170
"""

import pandas as pd
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.regularizers import L1L2
from math import sqrt
import numpy as np
import random as rn
import os
import sys


# convert time series into supervised learning problem
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


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# set base path
file_num = str(sys.argv[1]).split("/")[4].split(".")[0]
path = sys.argv[2]

# load dataset
dataset_raw = read_csv(sys.argv[1])
# dataset = read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_bootstraps/bs1.csv")
dataset = dataset_raw.drop(dataset_raw.columns[[0]], axis=1)

# configure network
n_lags = 48
n_ahead = 19
n_features = 3
n_train = round(len(dataset)*0.7)
n_test = len(dataset)-n_train
n_epochs = 10000
n_neurons = 75
n_batch = n_train

# split datetime column into train and test for plots
train_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[:n_train]
test_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[n_train:]
test_dates = test_dates.reset_index(drop=True)
test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

values = dataset[['GWL', 'Tide', 'Precip.']].values
values = values.astype('float32')

gwl = values[:, 0]
gwl = gwl.reshape(gwl.shape[0], 1)

tide = values[:, 1]
tide = tide.reshape(tide.shape[0], 1)

rain = values[:, 2]
rain = rain.reshape(rain.shape[0], 1)

# normalize features with individual scalers
gwl_scaler, tide_scaler, rain_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
gwl_scaled = gwl_scaler.fit_transform(gwl)
tide_scaled = tide_scaler.fit_transform(tide)
rain_scaled = rain_scaler.fit_transform(rain)

# scaled = np.concatenate((gwl_scaled, tide_scaled, rain_scaled), axis=1)

# frame as supervised learning
gwl_super = series_to_supervised(gwl_scaled, n_lags, n_ahead)
gwl_super_values = gwl_super.values
tide_super = series_to_supervised(tide_scaled, n_lags, n_ahead)
tide_super_values = tide_super.values
rain_super = series_to_supervised(rain_scaled, n_lags, n_ahead)
rain_super_values = rain_super.values

# split groundwater into inputs and labels
gwl_input, gwl_labels = gwl_super_values[:, 0:n_lags+1], gwl_super_values[:, n_lags+1:]

# split into train and test sets
train_X = np.concatenate((gwl_input[:n_train, :], tide_super_values[:n_train, :], rain_super_values[:n_train, :]),
                         axis=1)
test_X = np.concatenate((gwl_input[n_train:, :], tide_super_values[n_train:, :], rain_super_values[n_train:, :]),
                        axis=1)
train_y, test_y = gwl_labels[:n_train, :], gwl_labels[n_train:, :]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# set random seeds for model reproducibility as suggested in:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# define model
model = Sequential()
model.add(SimpleRNN(units=n_neurons, activation='tanh', input_shape=(None, train_X.shape[2]), use_bias=True,
                    bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=False))
model.add(Dropout(.145))
model.add(Dense(activation='linear', units=n_ahead-1, use_bias=True))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=rmse, optimizer=adam)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop])

# make predictions
trainPredict = model.predict(train_X)
yhat = model.predict(test_X)
inv_trainPredict = gwl_scaler.inverse_transform(trainPredict)
inv_yhat = gwl_scaler.inverse_transform(yhat)
inv_y = gwl_scaler.inverse_transform(test_y)
inv_train_y = gwl_scaler.inverse_transform(train_y)

# calculate RMSE for whole test series (each forecast step)
RMSE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mse = mean_squared_error(inv_y[:, j], inv_yhat[:, j])
    rmse = sqrt(mse)
    RMSE_forecast.append(rmse)
RMSE_forecast = DataFrame(RMSE_forecast)
rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average Test RMSE: %.3f' % rmse_avg)
RMSE_forecast.to_csv(os.path.join(path, file_num + "_RMSE.csv"))

# calculate NSE for each timestep ahead
NSE_forecast = []
for j in np.arange(0, inv_yhat.shape[1], 1):
    num_diff = np.subtract(inv_y[:, j], inv_yhat[:, j])
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(inv_y[:, j], np.mean(inv_y[:, j]))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1 - (numerator / denominator)
    NSE_forecast.append(nse)
NSE_forecast = DataFrame(NSE_forecast)
NSE_forecast.to_csv(os.path.join(path, file_num + "_NSE.csv"))

# calculate mean absolute error
MAE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mae = np.sum(np.abs(np.subtract(inv_y[:, j], inv_yhat[:, j]))) / inv_y.shape[0]
    MAE_forecast.append(mae)
MAE_forecast = DataFrame(MAE_forecast)
MAE_forecast.to_csv(os.path.join(path, file_num + "_MAE.csv"))

# create dfs of timestamps, obs, and pred data to find peak values and times
dates = DataFrame(test_dates[["Datetime"]][n_lags + 1:-n_ahead + 2])
dates = dates.reset_index(inplace=False, drop=True)
dates_9 = DataFrame(test_dates[["Datetime"]][n_lags + 9:-n_ahead + 10])
dates_9 = dates_9.reset_index(inplace=False, drop=True)
dates_18 = DataFrame(test_dates[["Datetime"]][n_lags + 18:])
dates_18 = dates_18.reset_index(inplace=False, drop=True)

obs_t1 = np.reshape(inv_y[:, 0], (inv_y.shape[0], 1))
pred_t1 = np.reshape(inv_yhat[:, 0], (inv_y.shape[0], 1))
df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
df_t1 = DataFrame(df_t1, index=None, columns=["obs", "pred"])
df_t1 = pd.concat([df_t1, dates], axis=1)
df_t1 = df_t1.set_index("Datetime")
df_t1 = df_t1.rename(columns={'obs': 'Obs. GWL t+1', 'pred': 'Pred. GWL t+1'})

obs_t9 = np.reshape(inv_y[:, 8], (inv_y.shape[0], 1))
pred_t9 = np.reshape(inv_yhat[:, 8], (inv_y.shape[0], 1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["obs", "pred"])
df_t9 = pd.concat([df_t9, dates_9], axis=1)
df_t9 = df_t9.set_index("Datetime")
df_t9 = df_t9.rename(columns={'obs': 'Obs. GWL t+9', 'pred': 'Pred. GWL t+9'})

obs_t18 = np.reshape(inv_y[:, 17], (inv_y.shape[0], 1))
pred_t18 = np.reshape(inv_yhat[:, 17], (inv_y.shape[0], 1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=["obs", "pred"])
df_t18 = pd.concat([df_t18, dates_18], axis=1)
df_t18 = df_t18.set_index("Datetime")
df_t18 = df_t18.rename(columns={'obs': 'Obs. GWL t+18', 'pred': 'Pred. GWL t+18'})

# combine prediction data with observations
if file_num == "bs0":
    test_dates = test_dates.set_index(pd.DatetimeIndex(test_dates['Datetime']))
    all_data_df = pd.concat([test_dates, df_t1[['Pred. GWL t+1']], df_t9[['Pred. GWL t+9']],
                             df_t18[['Pred. GWL t+18']]], axis=1)
    all_data_df.to_csv(os.path.join(path, file_num + "_all_data_df.csv"))
