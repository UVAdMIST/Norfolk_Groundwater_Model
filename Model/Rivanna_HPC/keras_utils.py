"""
Written by Benjamin Bowes, 22-10-2018

This script contains functions for data formatting and accuracy assessment of keras models
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from math import sqrt
import numpy as np


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# model cost function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# scale and format observed data as train/test inputs/labels
def format_obs_data(full_data, n_lags, n_ahead, n_train):
    # split datetime column into train and test for plots
    train_dates = full_data[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[:n_train]
    test_dates = full_data[['Datetime', 'GWL', 'Tide', 'Precip.']].iloc[n_train:]
    test_dates = test_dates.reset_index(drop=True)
    test_dates['Datetime'] = pd.to_datetime(test_dates['Datetime'])

    values = full_data[['GWL', 'Tide', 'Precip.']].values
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
    print("observed training input data shape:", train_X.shape, "observed training label data shape:", train_y.shape)
    print("observed testing input data shape:", test_X.shape, "observed testing label data shape:", test_y.shape)
    return train_dates, test_dates, tide_fit, rain_fit, gwl_fit, train_X, test_X, train_y, test_y


# scale and format storm data as train/test inputs/labels
def format_storm_data(storm_data, n_train, tide_fit, rain_fit, gwl_fit):
    # separate storm data into gwl, tide, and rain
    storm_scaled = pd.DataFrame(storm_data["Datetime"])
    for col in storm_data.columns:
        if col.split("(")[0] == "tide":
            col_data = np.asarray(storm_data[col])
            col_data = col_data.reshape(col_data.shape[0], 1)
            col_scaled = tide_fit.transform(col_data)
            storm_scaled[col] = col_scaled
        if col.split("(")[0] == "rain":
            col_data = np.asarray(storm_data[col])
            col_data = col_data.reshape(col_data.shape[0], 1)
            col_scaled = rain_fit.transform(col_data)
            storm_scaled[col] = col_scaled
        if col.split("(")[0] == "gwl":
            col_data = np.asarray(storm_data[col])
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
    print("observed training input data shape:", train_X.shape, "observed training label data shape:", train_y.shape)
    print("observed testing input data shape:", test_X.shape, "observed testing label data shape:", test_y.shape)
    return train_X, test_X, train_y, test_y


# scale and format forecast data as train/test inputs/labels
def format_fcst_data(fcst_data, tide_fit, rain_fit, gwl_fit):
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
    print("forecast input data shape:", fcst_test_X.shape, "forecast label data shape:", fcst_labels.shape)
    return fcst_test_X, fcst_labels


# create df of full observed data and predictions and extract storm data
def full_pred_df(test_dates, storm_data, n_lags, n_ahead, inv_y, inv_yhat):
    dates_t1 = pd.DataFrame(test_dates[["Datetime"]][n_lags + 1:-n_ahead + 2])
    dates_t1 = dates_t1.reset_index(inplace=False, drop=True)
    dates_9 = pd.DataFrame(test_dates[["Datetime"]][n_lags + 9:-n_ahead + 10])
    dates_9 = dates_9.reset_index(inplace=False, drop=True)
    dates_18 = pd.DataFrame(test_dates[["Datetime"]][n_lags + 18:])
    dates_18 = dates_18.reset_index(inplace=False, drop=True)

    obs_t1 = np.reshape(inv_y[:, 0], (inv_y.shape[0], 1))
    pred_t1 = np.reshape(inv_yhat[:, 0], (inv_y.shape[0], 1))
    df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
    df_t1 = pd.DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Pred. GWL t+1"])
    df_t1 = pd.concat([df_t1, dates_t1], axis=1)
    df_t1 = df_t1.set_index("Datetime")

    obs_t9 = np.reshape(inv_y[:, 8], (inv_y.shape[0], 1))
    pred_t9 = np.reshape(inv_yhat[:, 8], (inv_y.shape[0], 1))
    df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
    df_t9 = pd.DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Pred. GWL t+9"])
    df_t9 = pd.concat([df_t9, dates_9], axis=1)
    df_t9 = df_t9.set_index("Datetime")

    obs_t18 = np.reshape(inv_y[:, 17], (inv_y.shape[0], 1))
    pred_t18 = np.reshape(inv_yhat[:, 17], (inv_y.shape[0], 1))
    df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
    df_t18 = pd.DataFrame(df_t18, index=None, columns=["Obs. GWL t+18", "Pred. GWL t+18"])
    df_t18 = pd.concat([df_t18, dates_18], axis=1)
    df_t18 = df_t18.set_index("Datetime")

    storm_dates_t1 = storm_data[['gwl(t+1)']]
    storm_dates_t1.index = storm_dates_t1.index + pd.DateOffset(hours=1)

    storm_dates_t9 = storm_data[['gwl(t+9)']]
    storm_dates_t9.index = storm_dates_t9.index + pd.DateOffset(hours=9)

    storm_dates_t18 = storm_data[['gwl(t+18)']]
    storm_dates_t18.index = storm_dates_t18.index + pd.DateOffset(hours=18)

    df_t1_storms = np.asarray(df_t1[df_t1.index.isin(storm_dates_t1.index)])
    df_t9_storms = np.asarray(df_t9[df_t9.index.isin(storm_dates_t9.index)])
    df_t18_storms = np.asarray(df_t18[df_t18.index.isin(storm_dates_t18.index)])

    storms_list = [df_t1_storms, df_t9_storms, df_t18_storms]
    return df_t1, df_t9, df_t18, storms_list


# create df of storm observed data and predictions
def storm_pred_df(storm_data, n_train, inv_y, inv_yhat):
    test_dates_t1 = storm_data[['Datetime', 'tide(t+1)', 'rain(t+1)']].iloc[n_train:]
    test_dates_t1 = test_dates_t1.reset_index(drop=True)
    test_dates_t1['Datetime'] = pd.to_datetime(test_dates_t1['Datetime'])
    test_dates_t1['Datetime'] = test_dates_t1['Datetime'] + pd.DateOffset(hours=1)

    test_dates_t9 = storm_data[['Datetime', 'tide(t+9)', 'rain(t+9)']].iloc[n_train:]
    test_dates_t9 = test_dates_t9.reset_index(drop=True)
    test_dates_t9['Datetime'] = pd.to_datetime(test_dates_t9['Datetime'])
    test_dates_t9['Datetime'] = test_dates_t9['Datetime'] + pd.DateOffset(hours=9)

    test_dates_t18 = storm_data[['Datetime', 'tide(t+18)', 'rain(t+18)']].iloc[n_train:]
    test_dates_t18 = test_dates_t18.reset_index(drop=True)
    test_dates_t18['Datetime'] = pd.to_datetime(test_dates_t18['Datetime'])
    test_dates_t18['Datetime'] = test_dates_t18['Datetime'] + pd.DateOffset(hours=18)

    obs_t1 = np.reshape(inv_y[:, 0], (inv_y.shape[0], 1))
    pred_t1 = np.reshape(inv_yhat[:, 0], (inv_y.shape[0], 1))
    df_t1 = np.concatenate([obs_t1, pred_t1], axis=1)
    df_t1 = pd.DataFrame(df_t1, index=None, columns=["obs", "pred"])
    df_t1 = pd.concat([df_t1, test_dates_t1], axis=1)
    df_t1 = df_t1.set_index("Datetime")
    df_t1 = df_t1.rename(columns={'obs': 'Obs. GWL t+1', 'pred': 'Pred. GWL t+1'})

    obs_t9 = np.reshape(inv_y[:, 8], (inv_y.shape[0], 1))
    pred_t9 = np.reshape(inv_yhat[:, 8], (inv_y.shape[0], 1))
    df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
    df_t9 = pd.DataFrame(df_t9, index=None, columns=["obs", "pred"])
    df_t9 = pd.concat([df_t9, test_dates_t9], axis=1)
    df_t9 = df_t9.set_index("Datetime")
    df_t9 = df_t9.rename(columns={'obs': 'Obs. GWL t+9', 'pred': 'Pred. GWL t+9'})

    obs_t18 = np.reshape(inv_y[:, 17], (inv_y.shape[0], 1))
    pred_t18 = np.reshape(inv_yhat[:, 17], (inv_y.shape[0], 1))
    df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
    df_t18 = pd.DataFrame(df_t18, index=None, columns=["obs", "pred"])
    df_t18 = pd.concat([df_t18, test_dates_t18], axis=1)
    df_t18 = df_t18.set_index("Datetime")
    df_t18 = df_t18.rename(columns={'obs': 'Obs. GWL t+18', 'pred': 'Pred. GWL t+18'})

    all_data_df = pd.concat([df_t1, df_t9, df_t18], axis=1)
    return all_data_df


# create df of forecast data and predictions
def fcst_pred_df(fcst_data, inv_fcst_y, inv_fcst_yhat):
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
    df_t1 = pd.DataFrame(df_t1, index=None, columns=["Obs. GWL t+1", "Fcst. GWL t+1"])
    df_t1 = pd.concat([df_t1, test_dates_t1], axis=1)
    df_t1 = df_t1.set_index("Datetime")

    obs_t9 = np.reshape(inv_fcst_y[:, 8], (inv_fcst_y.shape[0], 1))
    pred_t9 = np.reshape(inv_fcst_yhat[:, 8], (inv_fcst_y.shape[0], 1))
    df_t9 = np.concatenate([obs_t9, pred_t9], axis=1)
    df_t9 = pd.DataFrame(df_t9, index=None, columns=["Obs. GWL t+9", "Fcst. GWL t+9"])
    df_t9 = pd.concat([df_t9, test_dates_t9], axis=1)
    df_t9 = df_t9.set_index("Datetime")

    obs_t18 = np.reshape(inv_fcst_y[:, 17], (inv_fcst_y.shape[0], 1))
    pred_t18 = np.reshape(inv_fcst_yhat[:, 17], (inv_fcst_y.shape[0], 1))
    df_t18 = np.concatenate([obs_t18, pred_t18], axis=1)
    df_t18 = pd.DataFrame(df_t18, index=None, columns=["Obs. GWL t+18", "Fcst. GWL t+18"])
    df_t18 = pd.concat([df_t18, test_dates_t18], axis=1)
    df_t18 = df_t18.set_index("Datetime")

    all_fcst_data_df = pd.concat([df_t1, df_t9, df_t18], axis=1)
    return all_fcst_data_df


# calculate model metrics
def calc_metrics(obs_y, pred_y, n_ahead):
    # calculate prediction RMSE
    rmse_list = []
    for j in np.arange(0, n_ahead - 1, 1):
        mse = mean_squared_error(obs_y[:, j], pred_y[:, j])
        rmse = sqrt(mse)
        rmse_list.append(rmse)
    rmse_list = pd.DataFrame(rmse_list)

    # calculate prediction MAE
    mae_list = []
    for j in np.arange(0, n_ahead - 1, 1):
        mae = np.sum(np.abs(np.subtract(obs_y[:, j], pred_y[:, j]))) / obs_y.shape[0]
        mae_list.append(mae)
    mae_list = pd.DataFrame(mae_list)

    # calculate prediction NSE
    nse_list = []
    for j in np.arange(0, pred_y.shape[1], 1):
        num_diff = np.subtract(obs_y[:, j], pred_y[:, j])
        num_sq = np.square(num_diff)
        numerator = sum(num_sq)
        denom_diff = np.subtract(obs_y[:, j], np.mean(obs_y[:, j]))
        denom_sq = np.square(denom_diff)
        denominator = sum(denom_sq)
        if denominator == 0:
            nse = 'NaN'
        else:
            nse = 1 - (numerator / denominator)
        nse_list.append(nse)
    nse_list = pd.DataFrame(nse_list)

    return rmse_list, mae_list, nse_list


# calculate performance of full model on storm data
def calc_metrics_fulldata_on_storms(storms_list):
    # calculate storm RMSE
    rmse_storms = []
    for j in storms_list:
        # print(j[:, 0])
        mse = mean_squared_error(j[:, 0], j[:, 1])
        single_rmse = sqrt(mse)
        rmse_storms.append(single_rmse)
    rmse_storms = pd.DataFrame(rmse_storms)

    # calculate storm NSE
    nse_storms = []
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
        nse_storms.append(nse)
    nse_storms = pd.DataFrame(nse_storms)

    # calculate storm MAE
    mae_storms = []
    for j in storms_list:
        mae = np.sum(np.abs(np.subtract(j[:, 0], j[:, 1]))) / j.shape[0]
        mae_storms.append(mae)
    mae_storms = pd.DataFrame(mae_storms)

    return rmse_storms, mae_storms, nse_storms
