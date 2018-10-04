""""
Written by Benjamin Bowes

This script creates supervised testing data sets with the following format:
    Observed input data: gwl(t-lag)...gwl(t), rain(t-lag)...rain(t), tide(t-lag)...tide(t)
    Forecast input data: rain(t+1)...rain(t+18), tide(t+1)...tide(t+18)
    Label data is still gwl(t+1)...gwl(t+18)
"""

import pandas as pd


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


obs_data_path = "C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/"
hrrr_dir = "C:/HRRR/"
fcst_path = "C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Forecast_data/"

# indicate which well to use
well_list = ["043", "125", "129", "153", "155", "170", "175"]
# well_list = ["043"]

n_ahead = 19

# read tide data
tide_2016 = pd.read_csv(fcst_path + "forecast_tide_raw_sept2016.csv", index_col="Date Time", parse_dates=True,
                        infer_datetime_format=True)
tide_2017 = pd.read_csv(fcst_path + "forecast_tide_raw_jan2017.csv", index_col="Date Time", parse_dates=True,
                        infer_datetime_format=True)
tide_2018 = pd.read_csv(fcst_path + "forecast_tide_raw_may2018.csv", index_col="Date Time", parse_dates=True,
                        infer_datetime_format=True)

# convert tide to supervised learning format and combine
tide2016_super = series_to_supervised(tide_2016, 0, n_ahead)
tide2016_super = tide2016_super.drop('var1(t)', axis=1)
tide2017_super = series_to_supervised(tide_2017, 0, n_ahead)
tide2017_super = tide2017_super.drop('var1(t)', axis=1)
tide2018_super = series_to_supervised(tide_2018, 0, n_ahead)
tide2018_super = tide2018_super.drop('var1(t)', axis=1)

forecast_tide_super = pd.concat([tide2016_super, tide2017_super, tide2018_super], axis=0)

for well in well_list:
    # lag and forecast values
    if well == "043":
        n_lag = 26
        rain_name = "Precip.Avg"
    if well == "125":
        n_lag = 26
        rain_name = "Precip."
    if well == "129":
        n_lag = 59
        rain_name = "Precip."
    if well == "153":
        n_lag = 25
        rain_name = "Precip.Avg"
    if well == "155":
        n_lag = 28
        rain_name = "Precip.Avg"
    if well == "170":
        n_lag = 48
        rain_name = "Precip."
    if well == "175":
        n_lag = 58
        rain_name = "Precip."

    # load observed and hrrr datasets
    obs_data = pd.read_csv(obs_data_path + "MMPS_" + well + "_no_blanks_SI.csv", parse_dates=True,
                           infer_datetime_format=True)

    hrrr_data = pd.read_csv(hrrr_dir + "mmps" + well + "_hrrr.csv", index_col="Datetime", parse_dates=True,
                            infer_datetime_format=True)

    # format obs data as supervised learning problem
    gwl_super = series_to_supervised(obs_data[["GWL"]], n_lag, n_ahead)
    gwl_cols = []
    for col in gwl_super.columns:
        col_name = "gwl(" + str(col).split("(")[1]
        gwl_cols.append(col_name)
    gwl_super.columns = gwl_cols
    gwl_dates = obs_data[obs_data.index.isin(gwl_super.index)]
    gwl_dates = gwl_dates[["Datetime"]]
    gwl_with_dates = pd.concat([gwl_dates, gwl_super], axis=1, sort=False)
    gwl_with_dates["Datetime"] = pd.to_datetime(gwl_with_dates["Datetime"])
    gwl_with_dates.set_index("Datetime", inplace=True)

    tide_super = series_to_supervised(obs_data[["Tide"]], n_lag, 1)
    tide_cols = []
    for col in tide_super.columns:
        col_name = "tide(" + str(col).split("(")[1]
        tide_cols.append(col_name)
    tide_super.columns = tide_cols
    tide_dates = obs_data[obs_data.index.isin(tide_super.index)]
    tide_dates = tide_dates[["Datetime"]]
    tide_with_dates = pd.concat([tide_dates, tide_super], axis=1, sort=False)
    tide_with_dates["Datetime"] = pd.to_datetime(tide_with_dates["Datetime"])
    tide_with_dates.set_index("Datetime", inplace=True)

    rain_super = series_to_supervised(obs_data[[rain_name]], n_lag, 1)
    rain_cols = []
    for col in rain_super.columns:
        col_name = "rain(" + str(col).split("(")[1]
        rain_cols.append(col_name)
    rain_super.columns = rain_cols
    rain_dates = obs_data[obs_data.index.isin(rain_super.index)]
    rain_dates = rain_dates[["Datetime"]]
    rain_with_dates = pd.concat([rain_dates, rain_super], axis=1, sort=False)
    rain_with_dates["Datetime"] = pd.to_datetime(rain_with_dates["Datetime"])
    rain_with_dates.set_index("Datetime", inplace=True)

    # get observed gwl, rain, and tide data that has same indices as hrrr
    gwl_super_hrrr = gwl_with_dates[gwl_with_dates.index.isin(hrrr_data.index)]
    tide_super_hrrr = tide_with_dates[tide_with_dates.index.isin(hrrr_data.index)]
    rain_super_hrrr = rain_with_dates[rain_with_dates.index.isin(hrrr_data.index)]

    # get forecast tide that has same indices as hrrr
    well_tide = forecast_tide_super[forecast_tide_super.index.isin(hrrr_data.index)]

    # make observed, hrrr, and tide datasets the same length
    hrrr_data = hrrr_data[:len(gwl_super_hrrr)]
    well_tide = well_tide[:len(gwl_super_hrrr)]
    tide_super_hrrr = tide_super_hrrr[:len(gwl_super_hrrr)]
    rain_super_hrrr = rain_super_hrrr[:len(gwl_super_hrrr)]

    # rename hrrr columns
    hrrr_cols = ["rain(t+1)", "rain(t+2)", "rain(t+3)", "rain(t+4)", "rain(t+5)", "rain(t+6)", "rain(t+7)", "rain(t+8)",
                 "rain(t+9)", "rain(t+10)", "rain(t+11)", "rain(t+12)", "rain(t+13)", "rain(t+14)", "rain(t+15)",
                 "rain(t+16)", "rain(t+17)", "rain(t+18)"]
    hrrr_data.columns = hrrr_cols

    # rename forecast tide columns
    fcst_tide_cols = []
    for col in well_tide.columns:
        col_name = "tide(" + str(col).split("(")[1]
        fcst_tide_cols.append(col_name)
    well_tide.columns = fcst_tide_cols

    # combine all data: obs tide, fcst tide, obs rain, fcst rain, gwl
    super_df = pd.concat([tide_super_hrrr, well_tide, rain_super_hrrr, hrrr_data, gwl_super_hrrr], axis=1)

    # save data
    super_df.to_csv(fcst_path + "MMPS_" + well + "testdata_SI.csv")
