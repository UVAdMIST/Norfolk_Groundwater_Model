"""
Written by Benjamin Bowes, 10-23-2018

Model: RNN
Train Data: storm data set, bootstrapped
Test Data: bootstrapped and forecast data

This network uses the last 48 observations of gwl, tide, and rain to predict the next 18
values of gwl for well MMPS-170
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.regularizers import L1L2
import random as rn
import os
import sys
from .keras_utils import *

# set base path
file_num = str(sys.argv[1]).split("/")[4].split(".")[0]
bs_path = sys.argv[2]
fcst_path = sys.argv[3]

# load storm dataset
storm_data = pd.read_csv(sys.argv[1])

# full dataset is needed to create scaler
full_data = pd.read_csv("/scratch/bdb3m/mmps170_bootstraps_shifted/bs0.csv", usecols=["Datetime", 'GWL', 'Tide', 'Precip.'])

# load forecast test data
fcst_data = pd.read_csv("/scratch/bdb3m/MMPS170_fcstdata_shifted_SI.csv")

# configure network
n_lags = 48
n_ahead = 19
n_features = 3
if file_num == "bs0":
    n_train = 26539
else:
    n_train = round(len(storm_data) * 0.7)
n_test = len(storm_data) - n_train
n_epochs = 10000
n_neurons = 75
n_batch = n_train

# format observed data to get scalers
_, _, tide_fit, rain_fit, gwl_fit, _, _, _, _ = format_obs_data(full_data, n_lags, n_ahead, n_train)

# format storm data
train_X, test_X, train_y, test_y = format_storm_data(storm_data, n_train, tide_fit, rain_fit, gwl_fit)

# format forecast data
fcst_test_X, fcst_labels = format_fcst_data(fcst_data, tide_fit, rain_fit, gwl_fit)

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

# make predictions with bootstrap test data
yhat = model.predict(test_X)
inv_yhat = gwl_fit.inverse_transform(yhat)
inv_y = gwl_fit.inverse_transform(test_y)

# make predictions with forecast test data
fcst_yhat = model.predict(fcst_test_X)
inv_fcst_yhat = gwl_fit.inverse_transform(fcst_yhat)
inv_fcst_y = gwl_fit.inverse_transform(fcst_labels)

# postprocess predictions to be <= land surface
inv_yhat[inv_yhat > 2.35] = 2.35
inv_fcst_yhat[inv_fcst_yhat > 2.35] = 2.35

# calc metrics for observed data
rmse_obs, mae_obs, nse_obs = calc_metrics(inv_y, inv_yhat, n_ahead)
rmse_obs.to_csv(os.path.join(bs_path, file_num + "_RMSE.csv"))
nse_obs.to_csv(os.path.join(bs_path, file_num + "_NSE.csv"))
mae_obs.to_csv(os.path.join(bs_path, file_num + "_MAE.csv"))

# calc metrics for forecast data
rmse_fcst, mae_fcst, nse_fcst = calc_metrics(inv_fcst_y, inv_fcst_yhat, n_ahead)
rmse_fcst.to_csv(os.path.join(fcst_path, file_num + "_RMSE.csv"))
nse_fcst.to_csv(os.path.join(fcst_path, file_num + "_NSE.csv"))
mae_fcst.to_csv(os.path.join(fcst_path, file_num + "_MAE.csv"))

if file_num == "bs0":
    # create df of storm observed data and predictions
    all_data_df = storm_pred_df(storm_data, n_train, inv_y, inv_yhat)
    all_data_df.to_csv(os.path.join(bs_path, file_num + "_all_data_df.csv"))

    # create df of forecast data and predictions
    all_fcst_data_df = fcst_pred_df(fcst_data, inv_fcst_y, inv_fcst_yhat)
    all_fcst_data_df.to_csv(os.path.join(fcst_path, file_num + "_all_fcst_data_df.csv"))
