"""
This network uses the last 26 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-043. Hyperparameters were chosen using the keras_mmps043_18hr_hyperas.py script
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
from keras.layers import Dense, SimpleRNN, LSTM
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


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


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


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1 - vec2)
            matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


# configure network
n_lags = 25
n_ahead = 19
n_features = 3
n_train = 56173
n_test = 17515
n_epochs = 10000
n_neurons = 75
n_batch = 56173

# set base path to store results
path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps153_results_rnn/"

# load dataset
dataset_raw = read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/MMPS_153_no_blanks_SI.csv",
                       index_col=None, parse_dates=True, infer_datetime_format=True)
# dataset_raw = dataset_raw[0:len(dataset_raw)-1]

# split datetime column into train and test for plots
train_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[:n_train]
test_dates = dataset_raw[['Datetime', 'GWL', 'Tide', 'Precip.Avg']].iloc[n_train:]
test_dates = test_dates.reset_index(drop=True)
test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

# drop columns we don't want to predict
dataset = dataset_raw.drop(dataset_raw.columns[[0, 3, 4, 5, 6]], axis=1)

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

# put data into form gwl, tide, rain, gwl, tide, rain, ...
# for i in range(gwl_super.shape[1]):


# split into train and test sets
train_X = np.concatenate((gwl_input[:n_train, :], tide_super_values[:n_train, :], rain_super_values[:n_train, :]),
                         axis=1)
test_X = np.concatenate((gwl_input[n_train:, :], tide_super_values[n_train:, :], rain_super_values[n_train:, :]),
                        axis=1)
train_y, test_y = gwl_labels[:n_train, :], gwl_labels[n_train:, :]

# save train and test data
np.savetxt(path + "train_X.csv", train_X, delimiter=',')
np.savetxt(path + "train_y.csv", train_y, delimiter=',')
np.savetxt(path + "test_X.csv", test_X, delimiter=',')
np.savetxt(path + "test_y.csv", test_y, delimiter=',')

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#create weights for peak weighted rmse loss function
# weights = create_weights(train_y)

# load model here if needed
# model = keras.models.load_model("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps043.h5",
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
model.add(SimpleRNN(units=n_neurons, activation='tanh', input_shape=(None, train_X.shape[2]), use_bias=True,
                    bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=False))  # Fully-connected RNN where the output is to be fed back to input.
# model.add(SimpleRNN(units=n_neurons, activation='tanh', use_bias=True, bias_regularizer=L1L2(l1=0.01, l2=0.01),
#                     return_sequences=True))
# model.add(SimpleRNN(units=n_neurons, activation='tanh', use_bias=True, bias_regularizer=L1L2(l1=0.01, l2=0.01)))
model.add(Dropout(.111))
model.add(Dense(activation='linear', units=n_ahead-1, use_bias=True))  # this is output layer
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=rmse, optimizer='adam')
tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/tmp/tensorflow/keras/logs', histogram_freq=0, write_graph=True,
                                         write_images=False)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop, tbCallBack])

# visualize model
# plot_model(model, to_file=path + "model_graph.png")

# save model
# model.save("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models/mmps043.h5")

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

# save scaled train predictions and observed
trainPredict_df = DataFrame(trainPredict)
trainPredict_df.to_csv(path + "scaled_train_predicted.csv")
train_y_df = DataFrame(train_y)
train_y_df.to_csv(path + "scaled_train_observed.csv")

# save train predictions and observed
inv_trainPredict_df = DataFrame(inv_trainPredict)
inv_trainPredict_df.to_csv(path + "train_predicted.csv")
inv_train_y_df = DataFrame(inv_train_y)
inv_train_y_df.to_csv(path + "train_observed.csv")

# save test predictions and observed
inv_yhat_df = DataFrame(inv_yhat)
inv_yhat_df.to_csv(path + "predicted.csv")
inv_y_df = DataFrame(inv_y)
inv_y_df.to_csv(path + "observed.csv")

# calculate RMSE for whole test series (each forecast step)
RMSE_forecast = []
MSE_forecast = []
for i in np.arange(0, n_ahead-1, 1):
    mse = mean_squared_error(inv_y[:, i], inv_yhat[:, i])
    rmse = sqrt(mse)
    RMSE_forecast.append(rmse)
    MSE_forecast.append(mse)
RMSE_forecast = DataFrame(RMSE_forecast)
MSE_forecast = DataFrame(MSE_forecast)
rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average Test RMSE: %.3f' % rmse_avg)
RMSE_forecast.to_csv(path + "RMSE.csv")
MSE_forecast.to_csv(path + "MSE.csv")

# calculate RMSE for each individual time step
RMSE_timestep = []
for i in np.arange(0, inv_yhat.shape[0], 1):
    rmse = sqrt(mean_squared_error(inv_y[i, :], inv_yhat[i, :]))
    RMSE_timestep.append(rmse)
RMSE_timestep = DataFrame(RMSE_timestep)

# plot rmse vs forecast steps
plt.plot(RMSE_forecast, 'ko')
ticks = np.arange(0, n_ahead-1, 1)  # (start,stop,increment)
plt.xticks(ticks)
plt.ylabel("RMSE (m)")
plt.xlabel("Forecast Step")
plt.tight_layout()
plt.show()

# plot training predictions
plt.plot(inv_train_y[:, 0], label='actual')
plt.plot(inv_trainPredict[:, 0], ':', label='predicted')
plt.xlabel("Timestep")
plt.ylabel("GWL (m)")
plt.title("Training Predictions")
# ticks = np.arange(0, n_ahead, 1)
# plt.xticks(ticks)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(path + "MMPS153_train_preds.pdf", dpi=300)
plt.close()

# plot test predictions for Hermine, Julia, and Matthew
dates = DataFrame(test_dates[["Datetime"]][n_lags+1:-n_ahead+2])
dates = dates.reset_index(inplace=False)
dates = dates.drop(columns=['index'])
dates_9 = DataFrame(test_dates[["Datetime"]][n_lags+9:-n_ahead+10])
dates_9 = dates_9.reset_index(inplace=False)
dates_9 = dates_9.drop(columns=['index'])
dates_18 = DataFrame(test_dates[["Datetime"]][n_lags+18:])
dates_18 = dates_18.reset_index(inplace=False)
dates_18 = dates_18.drop(columns=['index'])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
x_ticks = np.arange(0, inv_y.shape[0], 720)
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
ax1.text(-200, 2.5, 't+1')
ax2.text(-200, 2.5, 't+9')
ax3.text(-200, 2.5, 't+18')
ax3.set(ylabel="GWL (m)")
plt.legend(loc=9)
plt.tight_layout()
# plt.show()
fig.savefig(path + "MMPS153_forecast_preds.pdf", dpi=300)
plt.close()

# create dfs of timestamps, obs, and pred data to find peak values and times
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

# plot all test predictions
plt.plot(inv_y[:, 0], label='actual')
plt.plot(inv_yhat[:, 0], ':', label='predicted')
plt.xlabel("Timestep")
plt.ylabel("GWL (m)")
plt.title("Testing Predictions")
# ticks = np.arange(0, n_ahead, 1)
# plt.xticks(ticks)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(path + "MMPS153_alltest_preds.pdf", dpi=300)
plt.close()

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
all_data_df = pd.concat([test_dates, df_t1[['Pred. GWL t+1']], df_t9[['Pred. GWL t+9']], df_t18[['Pred. GWL t+18']]],
                        axis=1)

all_data_df.to_csv(path + "all_data_df.csv")

#create storm dfs
# Feb5Start, Feb5Stop = "2016-02-02T00:00:00.000000000", "2016-02-08T00:00:00.000000000"
# Jun6Start, Jun6Stop = "2016-06-05T00:00:00.000000000", "2016-06-12T00:00:00.000000000"
# Aug3Start, Aug3Stop = "2016-08-01T00:00:00.000000000", "2016-08-08T00:00:00.000000000"
HermineStart, HermineStop = "2016-09-01 00:00:00", "2016-09-07 00:00:00"
JuliaStart, JuliaStop = "2016-09-18 00:00:00", "2016-09-25 00:00:00"
MatthewStart, MatthewStop = "2016-10-07 00:00:00", "2016-10-12 00:00:00"

Sept_storms = all_data_df.loc[HermineStart:JuliaStop]
Sept_storms = Sept_storms.reset_index(drop=True)
Hermine = all_data_df.loc[HermineStart:HermineStop]
Hermine.name = "Hermine"
Julia = all_data_df.loc[JuliaStart:JuliaStop]
Julia.name = "Julia"
Matthew = all_data_df.loc[MatthewStart:MatthewStop]
Matthew = Matthew.reset_index(drop=True)

# determine peak values and times and RMSE/NSE for each storm and forecast
forecasts = ['1', '9', '18']
storms = [Hermine, Julia]

peaks = []
col_names = []
RMSE_pred = []
NSE_pred = []
for storm in storms:
    obs_peak = storm[['GWL']].max()
    obs_time = storm[['GWL']].idxmax()
    obs_peak_delta, obs_time_delta = 0, 0
    peaks.append([obs_peak[0], obs_peak_delta, obs_time[0], obs_time_delta])
    col_names.append(storm.name)
    for i in forecasts:
        forecast = "Pred. GWL t+" + i
        forecast_peak = storm[[forecast]].max()
        forecast_time = storm[[forecast]].idxmax()
        value_delta = obs_peak[0] - forecast_peak[0]
        time_delta = obs_time[0] - forecast_time[0]
        peaks.append([forecast_peak[0], value_delta, forecast_time[0], time_delta])
        col_names.append(storm.name + " t+" + i)

        rmse = sqrt(mean_squared_error(storm[['GWL']], storm[[forecast]]))
        RMSE_pred.append(rmse)

        num_diff = np.subtract(np.array(storm[['GWL']]), np.array(storm[[forecast]]))
        num_sq = np.square(num_diff)
        numerator = sum(num_sq)
        denom_diff = np.subtract(np.array(storm[['GWL']]), np.mean(np.array(storm[['GWL']])))
        denom_sq = np.square(denom_diff)
        denominator = sum(denom_sq)
        if denominator == 0:
            nse = 'NaN'
        else:
            nse = 1 - (numerator / denominator)
        NSE_pred.append(nse[0])

peaks_df = DataFrame(peaks, columns=['Peak Value', 'value delta', 'Peak Time', 'time delta'])
peaks_df = peaks_df.transpose()
peaks_df.columns = col_names
peaks_df.to_csv(path + "peaks.csv")

NSE_pred_df = DataFrame(NSE_pred, columns=["NSE"]).transpose()
RMSE_pred_df = DataFrame(RMSE_pred, columns=["RMSE"]).transpose()
storm_metrics_df = pd.concat([NSE_pred_df, RMSE_pred_df], axis=0)
storm_metrics_df.columns = ["Hermine t+1", "Hermine t+9", "Hermine t+18", "Julia t+1", "Julia t+9", "Julia t+18"]
storm_metrics_df.to_csv(path + "storm_metrics.csv")

Hermine = Hermine.reset_index(drop=True)
Julia = Julia.reset_index(drop=True)
Hermine.name = "Hermine"
Julia.name = "Julia"

# plot test predictions with observed rain and tide
forecasts = ['1', '9', '18']
storms = [Hermine, Julia]

for storm in storms:
    for i in forecasts:
        forecast = "Pred. GWL t+" + i
        ax = storm[["Tide", "GWL", forecast]].plot(color=["k"], style=[":", '-', '-.'], legend=None)
        start, end = ax.get_xlim()
        ticks = np.arange(0, end, 24)  # (start,stop,increment)
        ax2 = ax.twinx()
        ax2.set_ylim(ymax=50, ymin=0)
        ax.set_ylim(ymax=2.5, ymin=-0.5)
        ax2.invert_yaxis()
        storm["Precip.Avg"].plot.bar(ax=ax2, color="k")
        ax2.set_xticks([])
        ax.set_xticks(ticks)
        ax.set_xticklabels(storm.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
        ax.set_ylabel("Hourly Avg GW/Tide Level (m)")
        ax2.set_ylabel("Total Hourly Precip. (mm)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=1)  # location: 0=best, 9=top center
        plt.tight_layout()
        # plt.show()
        plot_path = path + "%s_t%s.pdf" % (storm.name, i)
        plt.savefig(plot_path, dpi=300)
        plt.close()

# save storm dataframes
Hermine.to_csv(path + "Hermine.csv")
Julia.to_csv(path + "Julia.csv")

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
NSE_forecast_df.to_csv(path + "NSE.csv")

# plot NSE vs forecast steps
plt.plot(NSE_forecast, 'ko')
ticks = np.arange(0, n_ahead, 1)  # (start,stop,increment)
plt.xticks(ticks)
plt.ylabel("NSE")
plt.xlabel("Forecast Step")
plt.tight_layout()
plt.show()

# calculate mean absolute error
MAE_forecast = []
for i in np.arange(0, n_ahead-1, 1):
    mae = np.sum(np.abs(np.subtract(inv_y[:, i], inv_yhat[:, i]))) / inv_y.shape[0]
    MAE_forecast.append(mae)
MAE_forecast = DataFrame(MAE_forecast)
MAE_forecast.to_csv(path + "MAE.csv")

# calculate DTW
# DTW_forecast = []
# for i in np.arange(0, n_ahead-1, 1):
#     matches, cost, mapping_1, mapping_2, matrix =dtw(inv_y[:, i], inv_yhat[:, i])
#     DTW_forecast.append(cost)
# DTW_forecast = DataFrame(DTW_forecast)
# DTW_forecast.to_csv(path + "DTW.csv")