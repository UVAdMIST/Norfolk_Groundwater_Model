import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from datetime import date, datetime, timedelta
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


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


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

col_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

mmps043_obs = mmps043_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps043_obs.columns = col_num
mmps125_obs = mmps125_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps125_obs.columns = col_num
mmps129_obs = mmps129_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps129_obs.columns = col_num
mmps153_obs = mmps153_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps153_obs.columns = col_num
mmps155_obs = mmps155_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps155_obs.columns = col_num
mmps170_obs = mmps170_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps170_obs.columns = col_num
mmps175_obs = mmps175_super.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
mmps175_obs.columns = col_num

# read HRRR data
hrrr_cols = ["Datetime", "f01", "f02", "f03", "f04","f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18"]
mmps043_hrrr = pd.read_csv("C:/HRRR/mmps043_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps043_hrrr.columns = col_num
mmps125_hrrr = pd.read_csv("C:/HRRR/mmps125_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps125_hrrr.columns = col_num
mmps129_hrrr = pd.read_csv("C:/HRRR/mmps129_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps129_hrrr.columns = col_num
mmps153_hrrr = pd.read_csv("C:/HRRR/mmps153_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps153_hrrr.columns = col_num
mmps155_hrrr = pd.read_csv("C:/HRRR/mmps155_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps155_hrrr.columns = col_num
mmps170_hrrr = pd.read_csv("C:/HRRR/mmps170_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps170_hrrr.columns = col_num
mmps175_hrrr = pd.read_csv("C:/HRRR/mmps175_hrrr.csv", index_col="Datetime", parse_dates=True, usecols=hrrr_cols)
mmps175_hrrr.columns = col_num

obs_list = [mmps043_obs, mmps125_obs, mmps129_obs, mmps153_obs, mmps155_obs, mmps170_obs, mmps175_obs]
hrrr_list = [mmps043_hrrr, mmps125_hrrr, mmps129_hrrr, mmps153_hrrr, mmps155_hrrr, mmps170_hrrr, mmps175_hrrr]
well_list = ["043", "125", "129", "153", "155", "170", "175"]

# plot obs vs hrrr for every hour
for i, j, k in zip(obs_list, hrrr_list, well_list):
    i = i.reset_index()
    j = j.reset_index()
    for m in range(0, 720, 1):
        date_list = []
        start_date = i.iloc[m][["Datetime"]]
        end_date = start_date + timedelta(hours=18)
        # dates = pd.date_range(start_date, end_date).tolist()
        for result in perdelta(start_date[0], end_date[0], timedelta(hours=1)):
            date_list.append(result)
        date_df = pd.DataFrame(date_list)
        date_df.columns = ["Datetime"]
        observed = pd.DataFrame(i.iloc[m, 1::1])
        observed.columns = ["Observed"]
        forecast = pd.DataFrame(j.iloc[m, 1::1])
        forecast.columns = ["Forecast"]
        df = pd.concat([date_df, observed, forecast], axis=1)
        ax = df[["Observed", "Forecast"]].plot.bar(color=['b', 'r'])
        start, end = ax.get_xlim()
        ticks = np.arange(0, end, 1)  # (start,stop,increment)
        ax.invert_yaxis()
        ax.set_xticks(ticks)
        ax.set_xticklabels(df.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d %H:%M'), rotation='vertical')
        if k == "043" or "125" or "129" or "170":
            ax.set_ylim(ymax=2.5, ymin=0)
        elif k == "155" or "175":
            ax.set_ylim(ymax=3.0, ymin=0)
        else:
            ax.set_ylim(ymax=3.7, ymin=0)
        ax.set_ylabel("Total Hourly Precip. (in)")
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + "/mmps" + k + "_plots" + "/" + str(m) + '.png', dpi=300)
        plt.close()
