"""
This script creates 1000 bootstrap replicates of time series data and stores them as csv files.
Columns to use for each well:

MMPS    Columns
043     [0, 1, 2, 7]
125     [0, 1, 2, 3]
129     [0, 1, 2, 3]
153     [0, 1, 2, 7]
155     [0, 1, 2, 5]
170     [0, 1, 2, 3]
175     [0, 1, 2, 3]
"""

import pandas as pd
import numpy as np
from arch.bootstrap import CircularBlockBootstrap


def bs_to_df(x):
    df = pd.DataFrame(x, columns=[["Datetime", "GWL", "Tide", "Precip."]])
    df = df.reset_index(drop=True)
    bs_df_list.append(df)


# set base path to store results
path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps175_bootstraps/"

# load dataset
dataset_raw = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/MMPS_175_no_blanks_SI.csv",
                          index_col=None, parse_dates=True, infer_datetime_format=True)

dataset_raw_np = np.array(dataset_raw)
dataset_np = dataset_raw_np[:, [0, 1, 2, 3]]

# set up bootstrap parameters
bootstrap = CircularBlockBootstrap(720, dataset_np)

bs_df_list = []
results = bootstrap.apply(bs_to_df, 1000)

count = 0
for bs in bs_df_list:
    if count % 25 == 0:
        print(count)
    f = path + "bs" + str(count) + ".csv"
    bs.to_csv(f)
    count += 1