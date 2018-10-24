"""
Model: LSTM
Train Data: storm data set
Test Data: storm data set and forecast data

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
import matplotlib.pyplot as plt
import matplotlib
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


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# set base path
results_path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/keras_models_17oct2018/mmps043_results_storm_lstm"
obs_path = os.path.join(results_path, "observed")
fcst_path = os.path.join(results_path, "forecast")

if not os.path.exists(results_path):
    os.makedirs(results_path)
    os.makedirs(obs_path)
    os.makedirs(fcst_path)
    print("created new directory: %s" % results_path)

# load dataset
# dataset = read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_bootstraps_storms_fixed/bs0.csv")
dataset = read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/Storm_data_fixed_22Oct2018/"
                   "MMPS_043_no_blanks_SI_storms_.csv")

# full dataset is needed to create scaler
full_dataset = read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_bootstraps/bs0.csv",
                        usecols=['Datetime', 'GWL', 'Tide', 'Precip.'])

# load forecast test data
fcst_data = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Forecast_data/MMPS043_fcstdata_SI.csv")

# configure network
n_lags = 26
n_ahead = 19
n_features = 3
# n_train = round(len(dataset)*0.7)
n_train = 13317
n_test = len(dataset)-n_train
n_epochs = 10000
n_neurons = 75
n_batch = n_train

values = full_dataset[['GWL', 'Tide', 'Precip.']].values
values = values.astype('float32')

gwl = values[:, 0]
gwl = gwl.reshape(gwl.shape[0], 1)

tide = values[:, 1]
tide = tide.reshape(tide.shape[0], 1)

rain = values[:, 2]
rain = rain.reshape(rain.shape[0], 1)

# normalize features with individual scalers
gwl_scaler, tide_scaler, rain_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
gwl_fit = gwl_scaler.fit(gwl)  # Compute the minimum and maximum gwl to be used for later scaling
tide_fit = tide_scaler.fit(tide)
rain_fit = rain_scaler.fit(rain)

# separate storm data into gwl, tide, and rain
storm_scaled = pd.DataFrame(dataset["Datetime"])
for col in dataset.columns:
    if col.split("(")[0] == "tide":
        col_data = np.asarray(dataset[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = tide_fit.transform(col_data)
        storm_scaled[col] = col_scaled
    if col.split("(")[0] == "rain":
        col_data = np.asarray(dataset[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = rain_fit.transform(col_data)
        storm_scaled[col] = col_scaled
    if col.split("(")[0] == "gwl":
        col_data = np.asarray(dataset[col])
        col_data = col_data.reshape(col_data.shape[0], 1)
        col_scaled = gwl_fit.transform(col_data)
        storm_scaled[col] = col_scaled

# split storm data into inputs and labels
storm_values = storm_scaled[storm_scaled.columns[1:]].values
storm_input, storm_labels = storm_values[:, :-18], storm_values[:, -18:]

# split into train and test sets
train_X, test_X = storm_input[:n_train, :], storm_input[n_train:, :]
train_y, test_y = storm_labels[:n_train, :], storm_labels[n_train:, :]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("observed data:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

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
print("forecast data:", fcst_test_X.shape, fcst_labels.shape)

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
#                bias_regularizer=L1L2(l1=0.01, l2=0.01)))  # This is hidden layer
model.add(Dropout(.355))
model.add(Dense(activation='linear', units=n_ahead-1, use_bias=True))  # this is output layer
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=rmse, optimizer=adam)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epochs, verbose=2, shuffle=False,
                    callbacks=[earlystop])

# plot model history
# plt.plot(history.history['loss'], label='train')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.tight_layout()
# plt.show()

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

# save scaled and unscaled forecast data and predictions
# np.savetxt("C:/Users/Ben Bowes/Desktop/mmps043_fcstdata_comparison/storm_fcst_yhat.csv", fcst_yhat, delimiter=',')
# np.savetxt("C:/Users/Ben Bowes/Desktop/mmps043_fcstdata_comparison/storm_inv_fcst_yhat.csv", inv_fcst_yhat, delimiter=',')
# np.savetxt("C:/Users/Ben Bowes/Desktop/mmps043_fcstdata_comparison/storm_fcst_y.csv", fcst_labels, delimiter=',')
# np.savetxt("C:/Users/Ben Bowes/Desktop/mmps043_fcstdata_comparison/storm_inv_fcst_y.csv", inv_fcst_y, delimiter=',')
# fcst_scaled.to_csv("C:/Users/Ben Bowes/Desktop/mmps043_fcstdata_comparison/storm_fcst_scaled.csv")

# calculate RMSE for observed data
RMSE_obs = []
for j in np.arange(0, n_ahead - 1, 1):
    mse = mean_squared_error(inv_y[:, j], inv_yhat[:, j])
    rmse = sqrt(mse)
    RMSE_obs.append(rmse)
RMSE_obs = DataFrame(RMSE_obs)
obs_rmse_avg = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Average bootstrap RMSE: %.3f' % obs_rmse_avg)
RMSE_obs.to_csv(os.path.join(obs_path, "RMSE.csv"))

# calculate RMSE for forecast data
RMSE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mse = mean_squared_error(inv_fcst_y[:, j], inv_fcst_yhat[:, j])
    rmse = sqrt(mse)
    RMSE_forecast.append(rmse)
RMSE_forecast = DataFrame(RMSE_forecast)
fcst_rmse_avg = sqrt(mean_squared_error(inv_fcst_y, inv_fcst_yhat))
print('Average forecast RMSE: %.3f' % fcst_rmse_avg)
RMSE_forecast.to_csv(os.path.join(fcst_path, "RMSE.csv"))

# calculate NSE for observed data
NSE_obs = []
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
    NSE_obs.append(nse)
NSE_obs = DataFrame(NSE_obs)
NSE_obs.to_csv(os.path.join(obs_path, "NSE.csv"))

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
NSE_forecast.to_csv(os.path.join(fcst_path, "NSE.csv"))

# calculate MAE for observed data
MAE_obs = []
for j in np.arange(0, n_ahead - 1, 1):
    mae = np.sum(np.abs(np.subtract(inv_y[:, j], inv_yhat[:, j]))) / inv_y.shape[0]
    MAE_obs.append(mae)
MAE_obs = DataFrame(MAE_obs)
MAE_obs.to_csv(os.path.join(obs_path, "MAE.csv"))

# calculate MAE for forecast data
MAE_forecast = []
for j in np.arange(0, n_ahead - 1, 1):
    mae = np.sum(np.abs(np.subtract(inv_fcst_y[:, j], inv_fcst_yhat[:, j]))) / inv_fcst_y.shape[0]
    MAE_forecast.append(mae)
MAE_forecast = DataFrame(MAE_forecast)
MAE_forecast.to_csv(os.path.join(fcst_path, "MAE.csv"))

# combine observed prediction data with observations
test_dates_t1 = dataset[['Datetime', 'tide(t+1)', 'rain(t+1)']].iloc[n_train:]
test_dates_t1 = test_dates_t1.reset_index(drop=True)
test_dates_t1['Datetime'] = pd.to_datetime(test_dates_t1['Datetime'])
test_dates_t1['Datetime'] = test_dates_t1['Datetime'] + pd.DateOffset(hours=1)

test_dates_t9 = dataset[['Datetime', 'tide(t+9)', 'rain(t+9)']].iloc[n_train:]
test_dates_t9 = test_dates_t9.reset_index(drop=True)
test_dates_t9['Datetime'] = pd.to_datetime(test_dates_t9['Datetime'])
test_dates_t9['Datetime'] = test_dates_t9['Datetime'] + pd.DateOffset(hours=9)

test_dates_t18 = dataset[['Datetime', 'tide(t+18)', 'rain(t+18)']].iloc[n_train:]
test_dates_t18 = test_dates_t18.reset_index(drop=True)
test_dates_t18['Datetime'] = pd.to_datetime(test_dates_t18['Datetime'])
test_dates_t18['Datetime'] = test_dates_t18['Datetime'] + pd.DateOffset(hours=18)

obs_t1 = np.reshape(inv_y[:, 0], (inv_y.shape[0], 1))
pred_t1 = np.reshape(inv_yhat[:, 0], (inv_y.shape[0], 1))
df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
df_t1 = DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Storm GWL t+1"])
df_t1 = pd.concat([df_t1, test_dates_t1], axis=1)
df_t1 = df_t1.set_index("Datetime")
df_t1 = df_t1.rename(columns={'tide(t+1)': 'obs tide(t+1)', 'rain(t+1)': 'obs rain(t+1)'})

obs_t9 = np.reshape(inv_y[:, 8], (inv_y.shape[0], 1))
pred_t9 = np.reshape(inv_yhat[:, 8], (inv_y.shape[0], 1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Storm GWL t+9"])
df_t9 = pd.concat([df_t9, test_dates_t9], axis=1)
df_t9 = df_t9.set_index("Datetime")
df_t9 = df_t9.rename(columns={'tide(t+9)': 'obs tide(t+9)', 'rain(t+9)': 'obs rain(t+9)'})

obs_t18 = np.reshape(inv_y[:, 17], (inv_y.shape[0], 1))
pred_t18 = np.reshape(inv_yhat[:, 17], (inv_y.shape[0], 1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=['Obs. GWL t+18', 'Storm GWL t+18'])
df_t18 = pd.concat([df_t18, test_dates_t18], axis=1)
df_t18 = df_t18.set_index("Datetime")
df_t18 = df_t18.rename(columns={'tide(t+18)': 'obs tide(t+18)', 'rain(t+18)': 'obs rain(t+18)'})

all_data_df = pd.concat([df_t1, df_t9, df_t18], axis=1)
all_data_df.to_csv(os.path.join(obs_path, "all_data_df.csv"))

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
df_t1 = DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Fcst. GWL t+1"])
df_t1 = pd.concat([df_t1, test_dates_t1], axis=1)
df_t1 = df_t1.set_index("Datetime")
df_t1 = df_t1.rename(columns={'tide(t+1)': 'fcst tide(t+1)', 'rain(t+1)': 'fcst rain(t+1)'})

obs_t9 = np.reshape(inv_fcst_y[:, 8], (inv_fcst_y.shape[0], 1))
pred_t9 = np.reshape(inv_fcst_yhat[:, 8], (inv_fcst_y.shape[0], 1))
df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
df_t9 = DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Fcst. GWL t+9"])
df_t9 = pd.concat([df_t9, test_dates_t9], axis=1)
df_t9 = df_t9.set_index("Datetime")
df_t9 = df_t9.rename(columns={'tide(t+9)': 'fcst tide(t+9)', 'rain(t+9)': 'fcst rain(t+9)'})

obs_t18 = np.reshape(inv_fcst_y[:, 17], (inv_fcst_y.shape[0], 1))
pred_t18 = np.reshape(inv_fcst_yhat[:, 17], (inv_fcst_y.shape[0], 1))
df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
df_t18 = DataFrame(df_t18, index=None, columns=["Obs. GWL t+18", "Fcst. GWL t+18"])
df_t18 = pd.concat([df_t18, test_dates_t18], axis=1)
df_t18 = df_t18.set_index("Datetime")
df_t18 = df_t18.rename(columns={'tide(t+18)': 'fcst tide(t+18)', 'rain(t+18)': 'fcst rain(t+18)'})

all_fcst_data_df = pd.concat([df_t1, df_t9, df_t18], axis=1)
all_fcst_data_df.to_csv(os.path.join(fcst_path, "all_fcst_data_df.csv"))

# combine obs and fcst data for plotting
storm_df = all_data_df[['Storm GWL t+1', 'obs tide(t+1)', 'obs rain(t+1)',  'Storm GWL t+9', 'obs tide(t+9)',
                        'obs rain(t+9)', 'Storm GWL t+18', 'obs tide(t+18)', 'obs rain(t+18)']]
plot_df = pd.concat([all_fcst_data_df, storm_df], axis=1)

# individual plots for each test period
starts = ["2016-08-31 21:00:00", "2016-12-30 21:00:00", "2018-04-30 21:00:00"]
stops = ["2016-10-01 13:00:00", "2017-02-01 13:00:00", "2018-05-31 08:00:00"]
for start, stop in zip(starts, stops):
    print(start, stop)
    period_data = plot_df.loc[start:stop].reset_index()

    fig, axs = plt.subplots(3, sharey=False, sharex=True, figsize=(6, 8))
    for plot_num, ax in enumerate(fig.axes):
        print(plot_num)
        if plot_num == 0:
            observed = "Obs. GWL t+1"
            obs_tide = "obs tide(t+1)"
            obs_rain = "obs rain(t+1)"
            storm_pred = "Storm GWL t+1"
            fcst_tide = "fcst tide(t+1)"
            fcst_rain = "fcst rain(t+1)"
            fcst_pred = "Fcst. GWL t+1"
            y1_label = ""
            y2_label = ""
        if plot_num == 1:
            observed = "Obs. GWL t+9"
            obs_tide = "obs tide(t+9)"
            obs_rain = "obs rain(t+9)"
            storm_pred = "Storm GWL t+9"
            fcst_tide = "fcst tide(t+9)"
            fcst_rain = "fcst rain(t+9)"
            fcst_pred = "Fcst. GWL t+9"
            y1_label = "Hourly Avg GW/Tide Level (m)"
            y2_label = "Total Hourly Precip. (mm)"
        if plot_num == 2:
            observed = "Obs. GWL t+18"
            obs_tide = "obs tide(t+18)"
            obs_rain = "obs rain(t+18)"
            storm_pred = "Storm GWL t+18"
            fcst_tide = "fcst tide(t+18)"
            fcst_rain = "fcst rain(t+18)"
            fcst_pred = "Fcst. GWL t+18"
            y1_label = ""
            y2_label = ""
        print(observed)

        ax = axs[plot_num]

        period_data[[observed, obs_tide, fcst_tide, storm_pred, fcst_pred]] \
            .plot(ax=ax, color=['k', 'k', 'r', 'b', 'r'], style=["-", ':', ':', '-.', '-.'],
                  label=["Obs. GWL", "Obs. Tide", "Fcst. Tide", "Pred. GWL Storm", "Pred. GWL Fcst."], legend=None)

        begin, end = ax.get_xlim()
        ticks = np.arange(0, end, 24)  # (start,stop,increment)
        ax2 = ax.twinx()
        # ax2.set_ylim(ymax=1, ymin=0)
        # ax.set_ylim(ymax=4.5, ymin=-1.5)
        ax2.invert_yaxis()
        period_data[[fcst_rain, obs_rain]].plot.bar(ax=ax2, color=['r', 'k'], legend=None)
        ax2.set_xticks([])
        ax.set_xticks(ticks)
        ax.set_xticklabels(period_data.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
        ax.set_ylabel(y1_label)
        # ax.set_ylim(ymax=13,ymin=-3)
        ax2.set_ylabel(y2_label)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center
    ax.legend(lines + lines2,
              ("Obs. GWL", "Obs. Tide", "Fcst. Tide", "Pred. GWL Storm", "Pred. GWL Fcst.", "Fcst. Precip.",
               "Obs. Precip."), bbox_to_anchor=(0.78, -0.38), ncol=2)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(results_path, "storm_comparison_" + str(plot_num + 1) + ".pdf"), dpi=300)
    # plt.close()

# make comparison plots for forecast data
# fig, axs = plt.subplots(3, 3, sharey='none', sharex='none', figsize=(8, 6))
# secondary_ax = []
# for plot_num, ax in enumerate(fig.axes):
#     print(plot_num)
#     if plot_num in [0, 1, 2]:
#         observed = "Obs. GWL t+1"
#         obs_tide = "obs tide(t+1)"
#         obs_rain = "obs rain(t+1)"
#         storm_pred = "Storm GWL t+1"
#         fcst_tide = "fcst tide(t+1)"
#         fcst_rain = "fcst rain(t+1)"
#         fcst_pred = "Fcst. GWL t+1"
#         row = 0
#     if plot_num in [3, 4, 5]:
#         observed = "Obs. GWL t+9"
#         obs_tide = "obs tide(t+9)"
#         obs_rain = "obs rain(t+9)"
#         storm_pred = "Storm GWL t+9"
#         fcst_tide = "fcst tide(t+9)"
#         fcst_rain = "fcst rain(t+9)"
#         fcst_pred = "Fcst. GWL t+9"
#         row = 1
#     if plot_num in [6, 7, 8]:
#         observed = "Obs. GWL t+18"
#         obs_tide = "obs tide(t+18)"
#         obs_rain = "obs rain(t+18)"
#         storm_pred = "Storm GWL t+18"
#         fcst_tide = "fcst tide(t+18)"
#         fcst_rain = "fcst rain(t+18)"
#         fcst_pred = "Fcst. GWL t+18"
#         row = 2
#
#     # create dfs of correct periods
#     if plot_num in [0, 3, 6]:
#         start, stop = "2016-08-31 21:00:00", "2016-10-01 13:00:00"
#         col = 0
#     if plot_num in [1, 4, 7]:
#         start, stop = "2016-12-30 21:00:00", "2017-02-01 13:00:00"
#         col = 1
#     if plot_num in [2, 5, 8]:
#         start, stop = "2018-04-30 21:00:00", "2018-05-31 23:00:00"
#         col = 2
#
#     if plot_num == 0:
#         y1_label = "t+1"
#     if plot_num == 3:
#         y1_label = "t+9"
#     if plot_num == 6:
#         y1_label = "t+18"
#     if plot_num in [1, 2, 4, 5, 7, 8]:
#         y1_label = ""
#
#     period_data = plot_df.loc[start:stop].reset_index()
#
#     ax = axs[row, col]
#     period_data[[observed, obs_tide, fcst_tide, storm_pred, fcst_pred]]\
#         .plot(ax=ax, color=['k', 'k', 'r', 'b', 'r'], style=["-", ':', ':', '-.', '-.'],
#               label=["Obs. GWL", "Obs. Tide", "Fcst. Tide", "Pred. GWL Storm", "Pred. GWL Fcst."], legend=None)
#     begin, end = ax.get_xlim()
#     ticks = np.arange(0, end, 48)  # (start,stop,increment)
#     ax2 = ax.twinx()
#     # ax2.set_ylim(ymax=1, ymin=0)
#     # ax.set_ylim(ymax=4.5, ymin=-1.5)
#     ax2.invert_yaxis()
#     period_data[[fcst_rain, obs_rain]].plot.bar(ax=ax2, color=['r', 'k'], legend=None)
#     ax2.set_xticks([])
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(period_data.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
#     ax.set_ylabel(y1_label)
#     # ax.set_ylim(ymax=13,ymin=-3)
#     # ax2.set_ylabel(y2_label)
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# # ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center
# plt.tight_layout()
# ax.legend(lines + lines2, ("Obs. GWL", "Obs. Tide", "Fcst. Tide", "Pred. GWL Storm", "Pred. GWL Fcst.", "Fcst. Precip.",
#                            "Obs. Precip."), bbox_to_anchor=(0.3, -0.33), ncol=7)
#
# fig.text(.01, .55, "Hourly Avg GW/Tide Level (m)", horizontalalignment='center', verticalalignment='center',
#          transform=ax.transAxes, rotation=90)
# fig.text(.99, .55, "Total Hourly Precip. (mm)", horizontalalignment='center', verticalalignment='center',
#          transform=ax.transAxes, rotation=-90)
# # plt.subplots_adjust(left=0.07)
# plt.show()
# # plt.savefig(os.path.join(results_path, "storm_comparison.pdf"), dpi=300)
# # plt.close()
