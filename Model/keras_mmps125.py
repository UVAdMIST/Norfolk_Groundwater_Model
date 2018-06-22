"""
This network uses the last 26 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-125. The data for MMPS-125 is missing Hurricane Matthew.
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
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import plot_model
from keras.regularizers import L1L2
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random as rn
import os
matplotlib.rcParams.update({'font.size': 8})


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


# def create_weights(train_labels):
#     obs_mean = np.mean(train_labels, axis=-1)
#     obs_mean = np.reshape(obs_mean, (n_batch, 1))
#     obs_mean = np.repeat(obs_mean, n_ahead, axis=1)
#     weights = (train_labels + obs_mean) / (2 * obs_mean)
#     return weights
#
#
# def sq_err(y_true, y_pred):
#     return K.square(y_pred - y_true)
#
#
def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def pw_rmse(y_true, y_pred):
    # num_rows, num_cols = K.int_shape(y_true)[0], K.int_shape(y_true)[1]
    # print(num_rows, num_cols)
    act_mean = K.mean(y_true, axis=-1)
    # print("act_mean 1 is:", act_mean)
    act_mean = K.reshape(act_mean, (n_batch, 1))
    # print("act_mean is: ", act_mean)
    mean_repeat = K.repeat_elements(act_mean, n_ahead, axis=1)
    # print("mean_repeat is:", mean_repeat)
    weights = (y_true+mean_repeat)/(2*mean_repeat)
    return K.sqrt(K.mean((K.square(y_pred - y_true)*weights), axis=-1))


# configure network
n_lags = 52
n_ahead = 18
n_features = 3
n_train = 48357
n_test = 7577
n_epochs = 10000
n_neurons = 10
n_batch = 48357

# load dataset
dataset_raw = read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_125_no_blanks.csv",
                       index_col=None, parse_dates=True, infer_datetime_format=True)

dataset_raw.loc[dataset_raw['GWL'] > 4.1, 'GWL'] = 4.1
# dataset_raw = dataset_raw[0:len(dataset_raw)-1]

# split datetime column into train and test for plots
train_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[:n_train]
test_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[n_train:]
test_dates = test_dates.reset_index(drop=True)
test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

# drop columns we don't want to predict
dataset = dataset_raw.drop(dataset_raw.columns[[0, 4]], axis=1)

values = dataset.values
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

scaled = np.concatenate((gwl_scaled, tide_scaled, rain_scaled), axis=1)

# frame as supervised learning
reframed = series_to_supervised(scaled, n_lags, n_ahead)
values = reframed.values

# split into train and test sets
train, test = values[:n_train, :], values[n_train:, :]
# split into input and outputs
input_cols, label_cols = [], []
for i in range(values.shape[1]):
    if i <= n_lags*n_features-1:
        input_cols.append(i)
    elif i % 3 != 0:
        input_cols.append(i)
    elif i % 3 == 0:
        label_cols.append(i)
train_X, train_y = train[:, input_cols], train[:, label_cols]  # [start:stop:increment, (cols to include)]
test_X, test_y = test[:, input_cols], test[:, label_cols]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#create weights for peak weighted rmse loss function
# weights = create_weights(train_y)

# load model here if needed
# model = keras.models.load_model("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps125.h5",
#                                 custom_objects={'pw_rmse':pw_rmse})

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
model.add(LSTM(units=n_neurons, input_shape=(None, train_X.shape[2]), use_bias=True,
               bias_regularizer=L1L2(l1=0.01, l2=0.01)))  # This is hidden layer
# model.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(None, train_X.shape[2]), use_bias=True))
# model.add(LSTM(units=n_neurons, return_sequences=True, use_bias=True))
# model.add(LSTM(units=n_neurons, use_bias=True))
model.add(Dropout(.1))
model.add(Dense(activation='linear', units=n_ahead, use_bias=True))  # this is output layer
# model.add(Activation('linear'))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=rmse, optimizer='adam')
tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/tmp/tensorflow/keras/logs', histogram_freq=0, write_graph=True,
                                         write_images=False)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=10, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop, tbCallBack])

# save model
# model.save("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps125.h5")

# plot model history
# plt.plot(history.history['loss'], label='train')
# # plt.plot(history.history['val_loss'], label='validate')
# # plt.legend()
# # ticks = np.arange(0, n_epochs, 1)  # (start,stop,increment)
# # plt.xticks(ticks)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.tight_layout()
# plt.show()

# make predictions
trainPredict = model.predict(train_X)
yhat = model.predict(test_X)
inv_trainPredict = gwl_scaler.inverse_transform(trainPredict)
inv_yhat = gwl_scaler.inverse_transform(yhat)
inv_y = gwl_scaler.inverse_transform(test_y)
inv_train_y = gwl_scaler.inverse_transform(train_y)

# post process predicted values to not be greater than the land surface elevation
inv_yhat[inv_yhat > 4.1] = 4.1

# save train predictions and observed
inv_trainPredict_df = DataFrame(inv_trainPredict)
inv_trainPredict_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/train_predicted.csv")
inv_train_y_df = DataFrame(inv_train_y)
inv_train_y_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/train_observed.csv")

# save test predictions and observed
inv_yhat_df = DataFrame(inv_yhat)
inv_yhat_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/predicted.csv")
inv_y_df = DataFrame(inv_y)
inv_y_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/observed.csv")

# calculate RMSE for whole test series (each forecast step)
RMSE_forecast = []
for i in np.arange(0, n_ahead, 1):
    rmse = sqrt(mean_squared_error(inv_y[:, i], inv_yhat[:, i]))
    RMSE_forecast.append(rmse)
RMSE_forecast = DataFrame(RMSE_forecast)
rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average Test RMSE: %.3f' % rmse_avg)
RMSE_forecast.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/RMSE.csv")

# calculate RMSE for each individual time step
RMSE_timestep = []
for i in np.arange(0, inv_yhat.shape[0], 1):
    rmse = sqrt(mean_squared_error(inv_y[i, :], inv_yhat[i, :]))
    RMSE_timestep.append(rmse)
RMSE_timestep = DataFrame(RMSE_timestep)

# plot rmse vs forecast steps
plt.plot(RMSE_forecast, 'ko')
ticks = np.arange(0, n_ahead, 1)  # (start,stop,increment)
plt.xticks(ticks)
plt.ylabel("RMSE (ft)")
plt.xlabel("Forecast Step")
plt.tight_layout()
plt.show()

# plot training predictions
plt.plot(inv_train_y[:, 0], label='actual')
plt.plot(inv_trainPredict[:, 0], label='predicted')
plt.xlabel("Timestep")
plt.ylabel("GWL (ft)")
plt.title("Training Predictions")
# ticks = np.arange(0, n_ahead, 1)
# plt.xticks(ticks)
plt.legend()
plt.tight_layout()
plt.show()

# plot test predictions for Hermine, Julia, and Matthew
dates = DataFrame(test_dates[["Datetime"]][n_lags:-n_ahead+1])
dates = dates.reset_index(inplace=False)
dates = dates.drop(columns=['index'])
dates = dates[:]
dates = dates.reset_index(inplace=False)
dates = dates.drop(columns=['index'])
dates_9 = DataFrame(test_dates[["Datetime"]][n_lags+8:-n_ahead+9])
dates_9 = dates_9.reset_index(inplace=False)
dates_9 = dates_9.drop(columns=['index'])
dates_9 = dates_9[:]
dates_9 = dates_9.reset_index(inplace=False)
dates_9 = dates_9.drop(columns=['index'])
dates_18 = DataFrame(test_dates[["Datetime"]][n_lags+17:])
dates_18 = dates_18.reset_index(inplace=False)
dates_18 = dates_18.drop(columns=['index'])
dates_18 = dates_18[:]
dates_18 = dates_18.reset_index(inplace=False)
dates_18 = dates_18.drop(columns=['index'])
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
x_ticks = np.arange(0, 7435, 720)
ax1.plot(inv_y[:, 0], '-', label='Obs.')
ax1.plot(inv_yhat[:, 0], ':', label='Pred.')
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(dates['Datetime'][x_ticks].dt.strftime('%m-%d'), rotation='vertical')
ax2.plot(inv_y[:, 8], '-', label='Obs.')
ax2.plot(inv_yhat[:, 8], ':', label='Pred.')
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(dates_9['Datetime'][x_ticks].dt.strftime('%m-%d'), rotation='vertical')
ax3.plot(inv_y[:, 17], '-', label='Obs.')
ax3.plot(inv_yhat[:, 17], ':', label='Pred.')
ax3.set_xticks(x_ticks)
ax3.set_xticklabels(dates_18['Datetime'][x_ticks].dt.strftime('%m-%d'), rotation='vertical')
ax1.text(-200, 4, 't+1')
ax2.text(-200, 4, 't+9')
ax3.text(-200, 4, 't+18')
ax2.set(ylabel="GWL (ft)")
plt.legend(loc=9)
plt.tight_layout()
plt.show()
fig.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Paper Figures/MMPS125_preds.tif', dpi=300)

# create dfs of timestamps, obs, and pred data to find peak values and times
obs_t1 = np.reshape(inv_y[:, 0], (7435, 1))
pred_t1 = np.reshape(inv_yhat[:, 0], (7435,1))
df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
df_t1 = DataFrame(df_t1, index=None, columns=["obs", "pred"])
df_t1 = pd.concat([df_t1, dates], axis=1)
df_t1 = df_t1.set_index("Datetime")
df_t1 = df_t1.rename(columns={'obs': 'Obs. GWL t+1', 'pred': 'Pred. GWL t+1'})

obs_t9 = np.reshape(inv_y[:, 8], (7435, 1))
pred_t9 = np.reshape(inv_yhat[:, 8], (7435,1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["obs", "pred"])
df_t9 = pd.concat([df_t9, dates_9], axis=1)
df_t9 = df_t9.set_index("Datetime")
df_t9 = df_t9.rename(columns={'obs': 'Obs. GWL t+9', 'pred': 'Pred. GWL t+9'})

obs_t18 = np.reshape(inv_y[:, 17], (7435, 1))
pred_t18 = np.reshape(inv_yhat[:, 17], (7435,1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=["obs", "pred"])
df_t18 = pd.concat([df_t18, dates_18], axis=1)
df_t18 = df_t18.set_index("Datetime")
df_t18 = df_t18.rename(columns={'obs': 'Obs. GWL t+18', 'pred': 'Pred. GWL t+18'})

Feb5Peak_t1 = df_t1.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].max()
Feb5Peak_t1_time = df_t1.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].idxmax()
Jun6Peak_t1 = df_t1.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].max()
Jun6Peak_t1_time = df_t1.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].idxmax()
Aug3Peak_t1 = df_t1.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].max()
Aug3Peak_t1_time = df_t1.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].idxmax()
HerminePeak_t1 = df_t1.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t1_time = df_t1.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t1 = df_t1.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t1_time = df_t1.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t1 = df_t1.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t1_time = df_t1.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

Feb5Peak_t9 = df_t9.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].max()
Feb5Peak_t9_time = df_t9.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].idxmax()
Jun6Peak_t9 = df_t9.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].max()
Jun6Peak_t9_time = df_t9.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].idxmax()
Aug3Peak_t9 = df_t9.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].max()
Aug3Peak_t9_time = df_t9.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].idxmax()
HerminePeak_t9 = df_t9.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t9_time = df_t9.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t9 = df_t9.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t9_time = df_t9.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t9 = df_t9.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t9_time = df_t9.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

Feb5Peak_t18 = df_t18.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].max()
Feb5Peak_t18_time = df_t18.loc["2016-02-02T00:00:00.000000000":"2016-02-08T00:00:00.000000000"].idxmax()
Jun6Peak_t18 = df_t18.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].max()
Jun6Peak_t18_time = df_t18.loc["2016-06-05T00:00:00.000000000":"2016-06-12T00:00:00.000000000"].idxmax()
Aug3Peak_t18 = df_t18.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].max()
Aug3Peak_t18_time = df_t18.loc["2016-08-01T00:00:00.000000000":"2016-08-08T00:00:00.000000000"].idxmax()
HerminePeak_t18 = df_t18.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t18_time = df_t18.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t18 = df_t18.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t18_time = df_t18.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t18 = df_t18.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t18_time = df_t18.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

peaks_values = DataFrame([Feb5Peak_t1, Jun6Peak_t1, Aug3Peak_t1, HerminePeak_t1, JuliaPeak_t1, MatthewPeak_t1,
                          Feb5Peak_t9, Jun6Peak_t9, Aug3Peak_t9, HerminePeak_t9, JuliaPeak_t9, MatthewPeak_t9,
                          Feb5Peak_t18, Jun6Peak_t18, Aug3Peak_t18, HerminePeak_t18, JuliaPeak_t18, MatthewPeak_t18])

peaks_values = peaks_values.transpose()
peaks_values.columns = ['Feb5Peak_t1', 'Jun6Peak_t1', 'Aug3Peak_t1', 'HerminePeak_t1', 'JuliaPeak_t1', 'MatthewPeak_t1',
                        'Feb5Peak_t9', 'Jun6Peak_t9', 'Aug3Peak_t9', 'HerminePeak_t9', 'JuliaPeak_t9', 'MatthewPeak_t9',
                        'Feb5Peak_t18', 'Jun6Peak_t18', 'Aug3Peak_t18', 'HerminePeak_t18', 'JuliaPeak_t18',
                        'MatthewPeak_t18']

peak_times = DataFrame([Feb5Peak_t1_time, Jun6Peak_t1_time, Aug3Peak_t1_time, HerminePeak_t1_time, JuliaPeak_t1_time,
                        MatthewPeak_t1_time, Feb5Peak_t9_time, Jun6Peak_t9_time, Aug3Peak_t9_time, HerminePeak_t9_time,
                        JuliaPeak_t9_time, MatthewPeak_t9_time, Feb5Peak_t18_time, Jun6Peak_t18_time, Aug3Peak_t18_time,
                        HerminePeak_t18_time, JuliaPeak_t18_time, MatthewPeak_t18_time])

peak_times = peak_times.transpose()
peak_times.columns = ['Feb5Time_t1', 'Jun6Time_t1', 'Aug3Time_t1', 'HermineTime_t1', 'JuliaTime_t1', 'MatthewTime_t1',
                      'Feb5Time_t9', 'Jun6Time_t9', 'Aug3Time_t9', 'HermineTime_t9', 'JuliaTime_t9', 'MatthewTime_t9',
                      'Feb5Time_t18', 'Jun6Time_t18', 'Aug3Time_t18', 'HermineTime_t18', 'JuliaTime_t18',
                      'MatthewTime_t18']

peaks_df = pd.concat([peaks_values, peak_times], axis=1)
cols = ['Feb5Peak_t1', 'Feb5Time_t1', 'Jun6Peak_t1', 'Jun6Time_t1', 'Aug3Peak_t1', 'Aug3Time_t1', 'HerminePeak_t1',
        'HermineTime_t1', 'JuliaPeak_t1', 'JuliaTime_t1', 'MatthewPeak_t1', 'MatthewTime_t1', 'Feb5Peak_t9',
        'Feb5Time_t9', 'Jun6Peak_t9', 'Jun6Time_t9', 'Aug3Peak_t9', 'Aug3Time_t9','HerminePeak_t9', 'HermineTime_t9',
        'JuliaPeak_t9', 'JuliaTime_t9', 'MatthewPeak_t9', 'MatthewTime_t9', 'Feb5Peak_t18', 'Feb5Time_t18',
        'Jun6Peak_t18', 'Jun6Time_t18', 'Aug3Peak_t18', 'Aug3Time_t18','HerminePeak_t18', 'HermineTime_t18',
        'JuliaPeak_t18', 'JuliaTime_t18', 'MatthewPeak_t18', 'MatthewTime_t18']
peaks_df = peaks_df[cols]
peaks_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/peaks.csv")

# plot all test predictions
plt.plot(inv_y[:, 0], label='actual')
plt.plot(inv_yhat[:, 0], label='predicted')
plt.xlabel("Timestep")
plt.ylabel("GWL (ft)")
plt.title("Testing Predictions")
# ticks = np.arange(0, n_ahead, 1)
# plt.xticks(ticks)
plt.legend()
plt.tight_layout()
plt.show()

# # plot test predictions, 18 hours from specific period
# plt.plot(inv_y[6275, :], label='actual')
# plt.plot(inv_yhat[6275, :], label='predicted')
# plt.xlabel("Timestep")
# plt.ylabel("GWL (ft)")
# plt.title("Testing Predictions")
# # ticks = np.arange(0, n_ahead, 1)
# # plt.xticks(ticks)
# plt.legend()
# plt.tight_layout()
# plt.show()

# combine prediction data with observations
test_dates = test_dates.set_index(pd.DatetimeIndex(test_dates['Datetime']))
all_data_df_t1 = pd.concat([df_t1, test_dates[['Tide', 'Precip.']][n_lags:-n_ahead+1]], axis=1)
all_data_df_t1 = all_data_df_t1.rename(columns={'obs': 'Obs. GWL', 'pred': 'Pred. GWL'})
all_data_df_t9 = pd.concat([df_t9, test_dates[['Tide', 'Precip.']][n_lags+8:-n_ahead+9]], axis=1)
all_data_df_t9 = all_data_df_t9.rename(columns={'obs': 'Obs. GWL', 'pred': 'Pred. GWL'})
all_data_df_t18 = pd.concat([df_t18, test_dates[['Tide', 'Precip.']][n_lags+17:]], axis=1)
all_data_df_t18 = all_data_df_t18.rename(columns={'obs': 'Obs. GWL', 'pred': 'Pred. GWL'})
all_data_df = pd.concat([test_dates, df_t1[['Pred. GWL t+1']], df_t9[['Pred. GWL t+9']], df_t18[['Pred. GWL t+18']]],
                        axis=1)

HermineStart, HermineStop = "2016-09-01 00:00:00", "2016-09-07 00:00:00"
JuliaStart, JuliaStop = "2016-09-18 00:00:00", "2016-09-25 00:00:00"
MatthewStart, MatthewStop = "2016-10-07 00:00:00", "2016-10-12 00:00:00"
Hermine = all_data_df.loc[HermineStart:HermineStop]
Hermine = Hermine.reset_index(drop=True)
Julia = all_data_df.loc[JuliaStart:JuliaStop]
Julia = Julia.reset_index(drop=True)
Matthew = all_data_df.loc[MatthewStart:MatthewStop]
Matthew = Matthew.reset_index(drop=True)

# plot test predictions with observed rain and tide
ax = Julia[["Tide", "GWL", "Pred. GWL t+18"]].plot(color=["k"], style=[":", '-', '-.'], legend=None)
start, end = ax.get_xlim()
ticks = np.arange(0, end, 24)  # (start,stop,increment)
ax2 = ax.twinx()
ax2.set_ylim(ymax=2.5, ymin=0)
ax.set_ylim(ymax=5, ymin=-1.25)
ax2.invert_yaxis()
Julia["Precip."].plot.bar(ax=ax2, color="k")
ax2.set_xticks([])
ax.set_xticks(ticks)
ax.set_xticklabels(Julia.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax.set_ylabel("Hourly Avg GW/Tide Level (ft)")
ax2.set_ylabel("Total Hourly Precip. (in)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1)  # location: 0=best, 9=top center
plt.tight_layout()
plt.show()
# save plot for publication
# plt.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Paper Figures/MMPS125_preds_allvars.tif', dpi=300)

# calculate NSE for each forecast period
NSE_timestep = []
for i in np.arange(0, inv_yhat.shape[0], 1):
    num_diff = np.subtract(inv_y[i, :], inv_yhat[i, :])
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(inv_y[i, :], np.mean(inv_y[i, :]))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1-(numerator/denominator)
    NSE_timestep.append(nse)
NSE_timestep_df = DataFrame(NSE_timestep)

# calculate NSE for each timestep ahead
NSE_forecast = []
for i in np.arange(0, inv_yhat.shape[1], 1):
    num_diff = np.subtract(inv_y[:, i], inv_yhat[:, i])
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(inv_y[:, i], np.mean(inv_y[:, i]))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1-(numerator/denominator)
    NSE_forecast.append(nse)
NSE_forecast_df = DataFrame(NSE_forecast)
NSE_forecast_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps125_results/NSE.csv")

# plot NSE vs forecast steps
plt.plot(NSE_forecast, 'ko')
ticks = np.arange(0, n_ahead, 1)  # (start,stop,increment)
plt.xticks(ticks)
plt.ylabel("NSE")
plt.xlabel("Forecast Step")
plt.tight_layout()
plt.show()
