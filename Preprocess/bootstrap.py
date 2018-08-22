"""
This script creates 1000 bootstrap replicates of time series data for each well
in Norfolk and stores them as csv files.
"""

import pandas as pd
import numpy as np
from arch.bootstrap import CircularBlockBootstrap


def bs_to_df(x):
    df = pd.DataFrame(x, columns=[["Datetime", "GWL", "Tide", "Precip."]])
    df = df.reset_index(drop=True)
    bs_df_list.append(df)


well_list = ["043", "125", "129", "153", "155", "170", "175"]

for well in well_list:
    # specify which columns and avg storm length to use for each well
    if well == "043":
        cols = [0, 1, 2, 7]
        storm_avg = 83
    if well == "125":
        cols = [0, 1, 2, 3]
        storm_avg = 82
    if well == "129":
        cols = [0, 1, 2, 3]
        storm_avg = 137
    if well == "153":
        cols = [0, 1, 2, 7]
        storm_avg = 89
    if well == "155":
        cols = [0, 1, 2, 5]
        storm_avg = 91
    if well == "170":
        cols = [0, 1, 2, 3]
        storm_avg = 120
    if well == "175":
        cols = [0, 1, 2, 3]
        storm_avg = 132

    # set base path to store results
    path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps" + well + "_bootstraps/"

    # load dataset
    dataset_raw = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/MMPS_" + well +
                              "_no_blanks_SI.csv", index_col=None, parse_dates=True, infer_datetime_format=True)

    dataset_raw_np = np.array(dataset_raw)
    dataset_np = dataset_raw_np[:, cols]

    # set up bootstrap parameters
    bootstrap = CircularBlockBootstrap(storm_avg, dataset_np)

    bs_df_list = []
    results = bootstrap.apply(bs_to_df, 1000)

    count = 0
    for bs in bs_df_list:
        if count % 25 == 0:
            print("well", well, "bootstrap:", count)
        f = path + "bs" + str(count) + ".csv"
        bs.to_csv(f)
        count += 1
