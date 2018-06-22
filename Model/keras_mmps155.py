"""
This network uses the last 26 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-155
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
from keras.layers import Dropout
from keras.layers import Activation
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
n_lags = 56
n_ahead = 18
n_features = 3
n_train = 52529
n_test = 8447
n_epochs = 500
n_neurons = 10
n_batch = 52529

# load dataset
dataset_raw = read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_155_no_blanks.csv",
                       index_col=None, parse_dates=True, infer_datetime_format=True)
# dataset_raw = dataset_raw[0:len(dataset_raw)-1]

# split datetime column into train and test for plots
train_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[:n_train]
test_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[n_train:]
test_dates = test_dates.reset_index(drop=True)
test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

# drop columns we don't want to predict
dataset = dataset_raw.drop(dataset_raw.columns[[0, 3, 4]], axis=1)

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
# model = keras.models.load_model("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps155.h5",
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
model.add(LSTM(units=n_neurons, input_shape=(None, train_X.shape[2])))
# model.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(None, train_X.shape[2])))
# model.add(LSTM(units=n_neurons, return_sequences=True))
# model.add(LSTM(units=n_neurons))
model.add(Dropout(.1))
model.add(Dense(input_dim=n_neurons, activation='linear', units=n_ahead))
# model.add(Activation('linear'))
model.compile(loss=pw_rmse, optimizer='adam')
tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/tmp/tensorflow/keras/logs', histogram_freq=0, write_graph=True,
                                         write_images=False)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop, tbCallBack])

# save model
# model.save("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps155.h5")

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

# save test predictions and observed
inv_yhat_df = DataFrame(inv_yhat)
inv_yhat_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps155_results/predicted.csv")
inv_y_df = DataFrame(inv_y)
inv_y_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps155_results/observed.csv")

# calculate RMSE for whole test series (each forecast step)
RMSE_forecast = []
for i in np.arange(0, n_ahead, 1):
    rmse = sqrt(mean_squared_error(inv_y[:, i], inv_yhat[:, i]))
    RMSE_forecast.append(rmse)
RMSE_forecast = DataFrame(RMSE_forecast)
rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average Test RMSE: %.3f' % rmse_avg)
RMSE_forecast.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps155_results/RMSE.csv")

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
dates = dates[5700:8000]
dates = dates.reset_index(inplace=False)
dates = dates.drop(columns=['index'])
dates_9 = DataFrame(test_dates[["Datetime"]][n_lags+8:-n_ahead+9])
dates_9 = dates_9.reset_index(inplace=False)
dates_9 = dates_9.drop(columns=['index'])
dates_9 = dates_9[5700:8000]
dates_9 = dates_9.reset_index(inplace=False)
dates_9 = dates_9.drop(columns=['index'])
dates_18 = DataFrame(test_dates[["Datetime"]][n_lags+17:])
dates_18 = dates_18.reset_index(inplace=False)
dates_18 = dates_18.drop(columns=['index'])
dates_18 = dates_18[5700:8000]
dates_18 = dates_18.reset_index(inplace=False)
dates_18 = dates_18.drop(columns=['index'])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(6.5, 3))
x_ticks = np.arange(0, 2300, 168)
ax1.plot(inv_y[5700:8000, 0], 'k-', label='Obs.')
ax1.plot(inv_yhat[5700:8000, 0], 'k:', label='Pred.')
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(dates['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax2.plot(inv_y[5700:8000, 8], 'k-', label='Obs.')
ax2.plot(inv_yhat[5700:8000, 8], 'k:', label='Pred.')
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(dates_9['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax3.plot(inv_y[5700:8000, 17], 'k-', label='Obs.')
ax3.plot(inv_yhat[5700:8000, 17], 'k:', label='Pred.')
ax3.set_xticks(x_ticks)
ax3.set_xticklabels(dates_18['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax1.set(ylabel="GWL (ft)", title='t+1')
ax2.set(title='t+9')
ax3.set(title='t+18')
plt.legend()
plt.tight_layout()
plt.show()
# fig.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Paper Figures/MMPS155_preds.tif', dpi=300)

# create dfs of timestamps, obs, and pred data to find peak values and times
obs_t1 = np.reshape(inv_y[5700:8000, 0], (2300, 1))
pred_t1 = np.reshape(inv_yhat[5700:8000, 0], (2300,1))
df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
df_t1 = DataFrame(df_t1, index=None, columns=["obs", "pred"])
df_t1 = pd.concat([df_t1, dates], axis=1)
df_t1 = df_t1.set_index("Datetime")

obs_t9 = np.reshape(inv_y[5700:8000, 8], (2300, 1))
pred_t9 = np.reshape(inv_yhat[5700:8000, 8], (2300,1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["obs", "pred"])
df_t9 = pd.concat([df_t9, dates_9], axis=1)
df_t9 = df_t9.set_index("Datetime")

obs_t18 = np.reshape(inv_y[5700:8000, 17], (2300, 1))
pred_t18 = np.reshape(inv_yhat[5700:8000, 17], (2300,1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=["obs", "pred"])
df_t18 = pd.concat([df_t18, dates_18], axis=1)
df_t18 = df_t18.set_index("Datetime")

HerminePeak_t1 = df_t1.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t1_time = df_t1.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t1 = df_t1.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t1_time = df_t1.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t1 = df_t1.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t1_time = df_t1.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

HerminePeak_t9 = df_t9.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t9_time = df_t9.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t9 = df_t9.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t9_time = df_t9.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t9 = df_t9.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t9_time = df_t9.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

HerminePeak_t18 = df_t18.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].max()
HerminePeak_t18_time = df_t18.loc["2016-09-02T00:00:00.000000000":"2016-09-08T00:00:00.000000000"].idxmax()
JuliaPeak_t18 = df_t18.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].max()
JuliaPeak_t18_time = df_t18.loc["2016-09-18T00:00:00.000000000":"2016-09-25T00:00:00.000000000"].idxmax()
MatthewPeak_t18 = df_t18.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].max()
MatthewPeak_t18_time = df_t18.loc["2016-10-07T00:00:00.000000000":"2016-10-14T00:00:00.000000000"].idxmax()

peaks_values = DataFrame([HerminePeak_t1, JuliaPeak_t1, MatthewPeak_t1, HerminePeak_t9, JuliaPeak_t9, MatthewPeak_t9,
                      HerminePeak_t18, JuliaPeak_t18, MatthewPeak_t18])

peaks_values = peaks_values.transpose()
peaks_values.columns = ['HerminePeak_t1', 'JuliaPeak_t1', 'MatthewPeak_t1', 'HerminePeak_t9', 'JuliaPeak_t9',
                        'MatthewPeak_t9', 'HerminePeak_t18', 'JuliaPeak_t18', 'MatthewPeak_t18']

peak_times = DataFrame([HerminePeak_t1_time, JuliaPeak_t1_time, MatthewPeak_t1_time, HerminePeak_t9_time,
                        JuliaPeak_t9_time, MatthewPeak_t9_time, HerminePeak_t18_time, JuliaPeak_t18_time,
                        MatthewPeak_t18_time])

peak_times = peak_times.transpose()
peak_times.columns = ['HermineTime_t1', 'JuliaTime_t1', 'MatthewTime_t1', 'HermineTime_t9', 'JuliaTime_t9',
                      'MatthewTime_t9', 'HermineTime_t18', 'JuliaTime_t18', 'MatthewTime_t18']

peaks_df = pd.concat([peaks_values, peak_times], axis=1)
cols = ['HerminePeak_t1', 'HermineTime_t1', 'JuliaPeak_t1', 'JuliaTime_t1', 'MatthewPeak_t1', 'MatthewTime_t1',
        'HerminePeak_t9', 'HermineTime_t9', 'JuliaPeak_t9', 'JuliaTime_t9', 'MatthewPeak_t9', 'MatthewTime_t9',
        'HerminePeak_t18', 'HermineTime_t18', 'JuliaPeak_t18', 'JuliaTime_t18', 'MatthewPeak_t18', 'MatthewTime_t18']
peaks_df = peaks_df[cols]
peaks_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps155_results/peaks.csv")

# plot all test predictions
plt.plot(inv_y[5700:8000, 17], label='actual')
plt.plot(inv_yhat[5700:8000, 17], label='predicted')
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
start_date, stop_date = "2016-09-17 00:00:00", "2016-09-25 00:00:00"
act_cols, pred_cols = [], []
for i in range(n_ahead):
    gwl_name = "Actual t+{0}".format(i)
    pred_name = 'Predicted t+{0}'.format(i)
    act_cols.append(gwl_name)
    pred_cols.append(pred_name)
df_act = DataFrame(inv_y, columns=act_cols)
df_pred = DataFrame(inv_yhat, columns=pred_cols)
df_gwl = pd.concat([df_act, df_pred], axis=1)
df = pd.concat([test_dates, df_gwl], axis=1)
df = df[:inv_y.shape[0]]
df = df.set_index('Datetime')
storm = df.loc[start_date:stop_date]
storm.reset_index(inplace=True)

# plot test predictions with observed rain and tide
ax = storm[["Tide", "Actual t+0", "Predicted t+0"]].plot(color=["k"], style=[":", '-', '-.'], legend=None)
start, end = ax.get_xlim()
ticks = np.arange(0, end, 24)  # (start,stop,increment)
ax2 = ax.twinx()
# ax2.set_ylim(ymax=2.5, ymin=0)
# ax.set_ylim(ymax=4, ymin=-1.25)
ax2.invert_yaxis()
storm["Precip.Avg"].plot.bar(ax=ax2, color="k")
ax2.set_xticks([])
ax.set_xticks(ticks)
ax.set_xticklabels(storm.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax.set_ylabel("Hourly Avg GW/Tide Level (ft)")
ax2.set_ylabel("Total Hourly Precip. (in)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1)  # location: 0=best, 9=top center
plt.tight_layout()
plt.show()
# save plot for publication
# plt.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Plots/Floods_GWL_comparisons/'
#             '20160919_bw_averaged.png', dpi=300)

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
NSE_forecast_df.to_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps155_results/NSE.csv")

# plot NSE vs forecast steps
plt.plot(NSE_forecast, 'ko')
ticks = np.arange(0, n_ahead, 1)  # (start,stop,increment)
plt.xticks(ticks)
plt.ylabel("NSE")
plt.xlabel("Forecast Step")
plt.tight_layout()
plt.show()
