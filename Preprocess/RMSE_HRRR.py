import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from datetime import datetime, timedelta
matplotlib.rcParams.update({'font.size': 8})


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


directory = "C:/HRRR"

# read observed data
MMPS_043 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_043_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip.Avg"])
MMPS_125 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_125_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip."])
MMPS_129 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_129_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip."])
MMPS_153 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_153_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip.Avg"])
MMPS_155 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_155_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip.Avg"])
MMPS_170 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_170_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip."])
MMPS_175 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_175_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True, usecols=["Datetime", "Precip."])

# create observed comparison data
mmps043_super = series_to_supervised(MMPS_043, 0, 18)
mmps125_super = series_to_supervised(MMPS_125, 0, 18)
mmps129_super = series_to_supervised(MMPS_129, 0, 18)
mmps153_super = series_to_supervised(MMPS_153, 0, 18)
mmps155_super = series_to_supervised(MMPS_155, 0, 18)
mmps170_super = series_to_supervised(MMPS_170, 0, 18)
mmps175_super = series_to_supervised(MMPS_175, 0, 18)

mmps043_obs = mmps043_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps125_obs = mmps125_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps129_obs = mmps129_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps153_obs = mmps153_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps155_obs = mmps155_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps170_obs = mmps170_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps175_obs = mmps175_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]

# read HRRR data
hrrr_cols = ["Datetime", "f01", "f02", "f03", "f04","f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18"]
mmps043_hrrr = pd.read_csv("C:/HRRR/mmps043_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps125_hrrr = pd.read_csv("C:/HRRR/mmps125_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps129_hrrr = pd.read_csv("C:/HRRR/mmps129_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps153_hrrr = pd.read_csv("C:/HRRR/mmps153_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps155_hrrr = pd.read_csv("C:/HRRR/mmps155_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps170_hrrr = pd.read_csv("C:/HRRR/mmps170_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps175_hrrr = pd.read_csv("C:/HRRR/mmps175_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)

obs_list = [mmps043_obs, mmps125_obs, mmps129_obs, mmps153_obs, mmps155_obs, mmps170_obs, mmps175_obs]
hrrr_list = [mmps043_hrrr, mmps125_hrrr, mmps129_hrrr, mmps153_hrrr, mmps155_hrrr, mmps170_hrrr, mmps175_hrrr]
well_list = ["043", "125", "129", "153", "155", "170", "175"]

# calculate RMSE for each data set
rmse_cols = ["f01", "f02", "f03", "f04","f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18", "avg"]

for i, j, k in zip(obs_list, hrrr_list, well_list):
    RMSE = []
    i = np.array(i)
    j = np.array(j)
    for m in np.arange(0, 18, 1):
        rmse = sqrt(mean_squared_error(i[:, m], j[:, m]))
        RMSE.append(rmse)
    rmse_avg = sqrt(mean_squared_error(i, j))
    RMSE.append(rmse_avg)
    RMSE_df = pd.DataFrame(RMSE).transpose()
    RMSE_df.columns = rmse_cols
    print('Average HRRR RMSE for MMPS-%s: %.3f' % (k, rmse_avg))
    RMSE_df.to_csv(directory + "/HRRR_RMSE_" + k + ".csv")
