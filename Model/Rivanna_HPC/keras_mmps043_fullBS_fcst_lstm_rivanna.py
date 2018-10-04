"""
Model: LSTM
Data: full data set, bootstrapped
Run from shell script
This network uses the last 26 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-043
"""

import pandas as pd
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
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
bs_path = sys.argv[2]
fcst_path = sys.argv[3]

# load dataset
dataset = read_csv(sys.argv[1])
# dataset = read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_bootstraps/bs0.csv")
dataset = dataset[['Datetime', 'GWL', 'Tide', 'Precip.']]

# load storm dataset to get indices for calculating performance on storms
storm_dataset = read_csv("/scratch/bdb3m/mmps043_bootstraps_storms_fixed/bs0.csv",
                         index_col="Datetime", parse_dates=True, infer_datetime_format=True,
                         usecols=['Datetime', 'gwl(t+1)', 'gwl(t+9)', 'gwl(t+18)'])
# storm_dataset = read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_bootstraps_storms_fixed/bs0.csv",
#                          index_col="Datetime", parse_dates=True, infer_datetime_format=True,
#                          usecols=['Datetime', 'gwl(t+1)', 'gwl(t+9)', 'gwl(t+18)'])

# load forecast test data
fcst_data = pd.read_csv("/scratch/bdb3m/MMPS043_fcstdata_SI.csv")
# fcst_data = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Forecast_data/MMPS043_fcstdata_SI.csv")

# configure network
n_lags = 26
n_ahead = 19
n_features = 3
n_train = round(len(dataset)*0.7)
n_test = len(dataset)-n_train
n_epochs = 10000
n_neurons = 75
n_batch = n_train

# split datetime column into train and test for plots
train_dates = dataset[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[:n_train]
test_dates = dataset[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[n_train:]
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
gwl_fit = gwl_scaler.fit(gwl)
gwl_scaled = gwl_fit.transform(gwl)
tide_fit = tide_scaler.fit(tide)
tide_scaled = tide_fit.transform(tide)
rain_fit = rain_scaler.fit(rain)
rain_scaled = rain_fit.transform(rain)

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
print("training data:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# separate forecast data into gwl, tide, and rain
fcst_scaled = pd.DataFrame(fcst_data["Datetime"])
for col in fcst_data.columns:
    if col.split("(")[0] == "tide":
        col_data = np.asarray(fcst_data[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = tide_fit.transform(col_data)
        fcst_scaled[col] = col_scaled
    if col.split("(")[0] == "rain":
        col_data = np.asarray(fcst_data[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = rain_fit.transform(col_data)
        fcst_scaled[col] = col_scaled
    if col.split("(")[0] == "gwl":
        col_data = np.asarray(fcst_data[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = gwl_fit.transform(col_data)
        fcst_scaled[col] = col_scaled

# split fcst data into inputs and labels
fcst_values = fcst_scaled[fcst_scaled.columns[1:]].values
fcst_input, fcst_labels = fcst_values[:, :-18], fcst_values[:, -18:]

# reshape fcst input to be 3D [samples, timesteps, features]
fcst_test_X = fcst_input.reshape((fcst_input.shape[0], 1, fcst_input.shape[1]))
print("testing data:", fcst_test_X.shape, fcst_labels.shape)

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
model.add(CuDNNLSTM(units=n_neurons, unit_forget_bias=True, bias_regularizer=L1L2(l1=0.01, l2=0.01)))
# model.add(LSTM(units=n_neurons, activation='tanh', input_shape=(None, train_X.shape[2]), use_bias=True,
#               bias_regularizer=L1L2(l1=0.01, l2=0.01)))  # This is hidden layer
model.add(Dropout(.355))
model.add(Dense(activation='linear', units=n_ahead-1, use_bias=True))  # this is output layer
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=rmse, optimizer=adam)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop])

# make predictions with bootstrap test data
yhat = model.predict(test_X)
inv_yhat = gwl_scaler.inverse_transform(yhat)
inv_y = gwl_scaler.inverse_transform(test_y)

# make predictions with forecast test data
fcst_yhat = model.predict(fcst_test_X)
inv_fcst_yhat = gwl_scaler.inverse_transform(fcst_yhat)
inv_fcst_y = gwl_scaler.inverse_transform(fcst_labels)

# postprocess predictions to be <= land surface
inv_yhat[inv_yhat > 2.21] = 2.21
inv_fcst_yhat[inv_fcst_yhat > 2.21] = 2.21

# calculate RMSE for bootstrap data
RMSE_bootstrap = []
for j in np.arange(0, n_ahead - 1, 1):
    mse = mean_squared_error(inv_y[:, j], inv_yhat[:, j])
    rmse = sqrt(mse)
    RMSE_bootstrap.append(rmse)
RMSE_bootstrap = DataFrame(RMSE_bootstrap)
bs_rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average bootstrap RMSE: %.3f' % bs_rmse_avg)
RMSE_bootstrap.to_csv(os.path.join(bs_path, file_num + "_RMSE.csv"))

# calculate RMSE for forecast data
RMSE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mse = mean_squared_error(inv_fcst_y[:, j], inv_fcst_yhat[:, j])
    rmse = sqrt(mse)
    RMSE_forecast.append(rmse)
RMSE_forecast = DataFrame(RMSE_forecast)
fcst_rmse_avg = sqrt(mean_squared_error(inv_fcst_y, inv_fcst_yhat))
print('Average forecast RMSE: %.3f' % fcst_rmse_avg)
RMSE_forecast.to_csv(os.path.join(fcst_path, file_num + "_RMSE.csv"))

# calculate NSE for bootstrap data
NSE_bootstrap = []
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
    NSE_bootstrap.append(nse)
NSE_bootstrap = DataFrame(NSE_bootstrap)
NSE_bootstrap.to_csv(os.path.join(bs_path, file_num + "_NSE.csv"))

# calculate NSE for forecast data
NSE_forecast = []
for j in np.arange(0, inv_fcst_yhat.shape[1], 1):
    num_diff = np.subtract(inv_fcst_y[:, j], inv_fcst_yhat[:, j])
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(inv_fcst_y[:, j], np.mean(inv_fcst_y[:, j]))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1 - (numerator / denominator)
    NSE_forecast.append(nse)
NSE_forecast = DataFrame(NSE_forecast)
NSE_forecast.to_csv(os.path.join(fcst_path, file_num + "_NSE.csv"))

# calculate MAE for bootstrap data
MAE_bootstrap = []
for j in np.arange(0, n_ahead - 1, 1):
    mae = np.sum(np.abs(np.subtract(inv_y[:, j], inv_yhat[:, j]))) / inv_y.shape[0]
    MAE_bootstrap.append(mae)
MAE_bootstrap = DataFrame(MAE_bootstrap)
MAE_bootstrap.to_csv(os.path.join(bs_path, file_num + "_MAE.csv"))

# calculate MAE for forecast data
MAE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mae = np.sum(np.abs(np.subtract(inv_fcst_y[:, j], inv_fcst_yhat[:, j]))) / inv_fcst_y.shape[0]
    MAE_forecast.append(mae)
MAE_forecast = DataFrame(MAE_forecast)
MAE_forecast.to_csv(os.path.join(fcst_path, file_num + "_MAE.csv"))

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
df_t1 = DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Pred. GWL t+1"])
df_t1 = pd.concat([df_t1, dates], axis=1)
df_t1 = df_t1.set_index("Datetime")

obs_t9 = np.reshape(inv_y[:, 8], (inv_y.shape[0], 1))
pred_t9 = np.reshape(inv_yhat[:, 8], (inv_y.shape[0], 1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Pred. GWL t+9"])
df_t9 = pd.concat([df_t9, dates_9], axis=1)
df_t9 = df_t9.set_index("Datetime")

obs_t18 = np.reshape(inv_y[:, 17], (inv_y.shape[0], 1))
pred_t18 = np.reshape(inv_yhat[:, 17], (inv_y.shape[0], 1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=["Obs. GWL t+18", "Pred. GWL t+18"])
df_t18 = pd.concat([df_t18, dates_18], axis=1)
df_t18 = df_t18.set_index("Datetime")

# extract storm dates from testset
storm_dates_t1 = storm_dataset[['gwl(t+1)']]
storm_dates_t1.index = storm_dates_t1.index + pd.DateOffset(hours=1)

storm_dates_t9 = storm_dataset[['gwl(t+9)']]
storm_dates_t9.index = storm_dates_t9.index + pd.DateOffset(hours=9)

storm_dates_t18 = storm_dataset[['gwl(t+18)']]
storm_dates_t18.index = storm_dates_t18.index + pd.DateOffset(hours=18)

df_t1_storms = np.asarray(df_t1[df_t1.index.isin(storm_dates_t1.index)])
df_t9_storms = np.asarray(df_t9[df_t9.index.isin(storm_dates_t9.index)])
df_t18_storms = np.asarray(df_t18[df_t18.index.isin(storm_dates_t18.index)])

storms_list = [df_t1_storms, df_t9_storms, df_t18_storms]

# calculate storm RMSE
RMSE_storms = []
for j in storms_list:
    # print(j[:, 0])
    mse = mean_squared_error(j[:, 0], j[:, 1])
    single_rmse = sqrt(mse)
    RMSE_storms.append(single_rmse)
RMSE_storms = DataFrame(RMSE_storms)
RMSE_storms.to_csv(os.path.join(bs_path, file_num + "_RMSE_storms.csv"))

# calculate storm NSE
NSE_storms = []
for j in storms_list:
    num_diff = np.subtract(j[:, 0], j[:, 1])
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(j[:, 0], np.mean(j[:, 0]))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1 - (numerator / denominator)
    NSE_storms.append(nse)
NSE_storms = DataFrame(NSE_storms)
NSE_storms.to_csv(os.path.join(bs_path, file_num + "_NSE_storms.csv"))

# calculate storm MAE
MAE_storms = []
for j in storms_list:
    mae = np.sum(np.abs(np.subtract(j[:, 0], j[:, 1]))) / j.shape[0]
    MAE_storms.append(mae)
MAE_storms = DataFrame(MAE_storms)
MAE_storms.to_csv(os.path.join(bs_path, file_num + "_MAE_storms.csv"))

# combine bootstrap prediction data with observations
if file_num == "bs0":
    test_dates = test_dates.set_index(pd.DatetimeIndex(test_dates['Datetime']))
    all_bs_data_df = pd.concat([test_dates, df_t1[['Pred. GWL t+1']], df_t9[['Pred. GWL t+9']],
                               df_t18[['Pred. GWL t+18']]], axis=1)
    all_bs_data_df.to_csv(os.path.join(bs_path, file_num + "_all_data_df.csv"))

    # combine forecast prediction data with observations
    test_dates_t1 = fcst_data[['Datetime', 'tide(t+1)', 'rain(t+1)']]
    test_dates_t1 = test_dates_t1.reset_index(drop=True)
    test_dates_t1['Datetime'] = pd.to_datetime(test_dates_t1['Datetime'])
    test_dates_t1['Datetime'] = test_dates_t1['Datetime'] + pd.DateOffset(hours=1)

    test_dates_t9 = fcst_data[['Datetime', 'tide(t+9)', 'rain(t+9)']]
    test_dates_t9 = test_dates_t9.reset_index(drop=True)
    test_dates_t9['Datetime'] = pd.to_datetime(test_dates_t9['Datetime'])
    test_dates_t9['Datetime'] = test_dates_t9['Datetime'] + pd.DateOffset(hours=9)

    test_dates_t18 = fcst_data[['Datetime', 'tide(t+18)', 'rain(t+18)']]
    test_dates_t18 = test_dates_t18.reset_index(drop=True)
    test_dates_t18['Datetime'] = pd.to_datetime(test_dates_t18['Datetime'])
    test_dates_t18['Datetime'] = test_dates_t18['Datetime'] + pd.DateOffset(hours=18)

    obs_t1 = np.reshape(inv_fcst_y[:, 0], (inv_fcst_y.shape[0], 1))
    pred_t1 = np.reshape(inv_fcst_yhat[:, 0], (inv_fcst_y.shape[0], 1))
    df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
    df_t1 = DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Pred. GWL t+1"])
    df_t1 = pd.concat([df_t1, test_dates_t1], axis=1)
    df_t1 = df_t1.set_index("Datetime")

    obs_t9 = np.reshape(inv_fcst_y[:, 8], (inv_fcst_y.shape[0], 1))
    pred_t9 = np.reshape(inv_fcst_yhat[:, 8], (inv_fcst_y.shape[0], 1))
    df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
    df_t9 = DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Pred. GWL t+9"])
    df_t9 = pd.concat([df_t9, test_dates_t9], axis=1)
    df_t9 = df_t9.set_index("Datetime")

    obs_t18 = np.reshape(inv_fcst_y[:, 17], (inv_fcst_y.shape[0], 1))
    pred_t18 = np.reshape(inv_fcst_yhat[:, 17], (inv_fcst_y.shape[0], 1))
    df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
    df_t18 = DataFrame(df_t18, index=None, columns=["Obs. GWL t+18", "Pred. GWL t+18"])
    df_t18 = pd.concat([df_t18, test_dates_t18], axis=1)
    df_t18 = df_t18.set_index("Datetime")

    all_fcst_data_df = pd.concat([df_t1, df_t9, df_t18], axis=1)
    all_fcst_data_df.to_csv(os.path.join(fcst_path, file_num + "_all_fcst_data_df.csv"))
