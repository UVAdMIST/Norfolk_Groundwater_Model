import pandas as pd
import os

directory = "C:/HRRR"

observed_data = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_043_no_blanks.csv",
                            index_col="Datetime", parse_dates=True, infer_datetime_format=True)

# create blank dataframes
mmps043_hrrr = pd.DataFrame()
mmps125_hrrr = pd.DataFrame()
mmps129_hrrr = pd.DataFrame()
mmps153_hrrr = pd.DataFrame()
mmps155_hrrr = pd.DataFrame()
mmps170_hrrr = pd.DataFrame()
mmps175_hrrr = pd.DataFrame()

for filename in os.listdir(directory):
    if filename.startswith("HRRR_Archive_2016"):
        f = os.path.join(directory, filename)
        hrrr_data = pd.read_csv(f + "/rainfall.csv", index_col="# Forecast")
        hrrr_data = hrrr_data[1:]
        data_043 = hrrr_data[["MMPS-043"]].transpose()
        mmps043_hrrr = pd.concat([mmps043_hrrr, data_043], axis=0)
        data_125 = hrrr_data[["MMPS-125"]].transpose()
        mmps125_hrrr = pd.concat([mmps125_hrrr, data_125], axis=0)
        data_129 = hrrr_data[["MMPS-129"]].transpose()
        mmps129_hrrr = pd.concat([mmps129_hrrr, data_129], axis=0)
        data_153 = hrrr_data[["MMPS-153"]].transpose()
        mmps153_hrrr = pd.concat([mmps153_hrrr, data_153], axis=0)
        data_155 = hrrr_data[["MMPS-155"]].transpose()
        mmps155_hrrr = pd.concat([mmps155_hrrr, data_155], axis=0)
        data_170 = hrrr_data[["MMPS-170"]].transpose()
        mmps170_hrrr = pd.concat([mmps170_hrrr, data_170], axis=0)
        data_175 = hrrr_data[["MMPS-175"]].transpose()
        mmps175_hrrr = pd.concat([mmps175_hrrr, data_175], axis=0)
        # for i in df_list:
            # print(i)
            # n_well = "MMPS-" + i.name
            # data = hrrr_data[[n_well]].transpose()
            # pd.concat([i, data], axis=0)
    else: continue

# rename columns
cols = ["f01", "f02", "f03", "f04","f05", "f06", "f07", "f08", "f09", "f10",
        "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18"]

mmps043_hrrr.columns = cols
mmps043_hrrr = mmps043_hrrr.reset_index(drop=True)
mmps125_hrrr.columns = cols
mmps125_hrrr = mmps125_hrrr.reset_index(drop=True)
mmps129_hrrr.columns = cols
mmps129_hrrr = mmps129_hrrr.reset_index(drop=True)
mmps153_hrrr.columns = cols
mmps153_hrrr = mmps153_hrrr.reset_index(drop=True)
mmps155_hrrr.columns = cols
mmps155_hrrr = mmps155_hrrr.reset_index(drop=True)
mmps170_hrrr.columns = cols
mmps170_hrrr = mmps170_hrrr.reset_index(drop=True)
mmps175_hrrr.columns = cols
mmps175_hrrr = mmps175_hrrr.reset_index(drop=True)

# add datetime
sept_dates = observed_data.loc["2016-08-31 21:00:00":"2016-09-30 20:00:00"]
sept_dates = sept_dates.reset_index()
mmps043_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps043_hrrr = mmps043_hrrr.set_index(pd.DatetimeIndex(mmps043_hrrr['Datetime']), drop=True)
mmps125_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps125_hrrr = mmps125_hrrr.set_index(pd.DatetimeIndex(mmps125_hrrr['Datetime']), drop=True)
mmps129_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps129_hrrr = mmps129_hrrr.set_index(pd.DatetimeIndex(mmps129_hrrr['Datetime']), drop=True)
mmps153_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps153_hrrr = mmps153_hrrr.set_index(pd.DatetimeIndex(mmps153_hrrr['Datetime']), drop=True)
mmps155_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps155_hrrr = mmps155_hrrr.set_index(pd.DatetimeIndex(mmps155_hrrr['Datetime']), drop=True)
mmps170_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps170_hrrr = mmps170_hrrr.set_index(pd.DatetimeIndex(mmps170_hrrr['Datetime']), drop=True)
mmps175_hrrr["Datetime"] = sept_dates[["Datetime"]]
mmps175_hrrr = mmps175_hrrr.set_index(pd.DatetimeIndex(mmps175_hrrr['Datetime']), drop=True)

# save dataframes
mmps043_hrrr.to_csv("C:/HRRR/mmps043_hrrr.csv")
mmps125_hrrr.to_csv("C:/HRRR/mmps125_hrrr.csv")
mmps129_hrrr.to_csv("C:/HRRR/mmps129_hrrr.csv")
mmps153_hrrr.to_csv("C:/HRRR/mmps153_hrrr.csv")
mmps155_hrrr.to_csv("C:/HRRR/mmps155_hrrr.csv")
mmps170_hrrr.to_csv("C:/HRRR/mmps170_hrrr.csv")
mmps175_hrrr.to_csv("C:/HRRR/mmps175_hrrr.csv")
