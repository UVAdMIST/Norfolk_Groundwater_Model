import pandas as pd
import sqlite3
import numpy as np


def hampel_filter(df, col, k, threshold=1):
    df['rolling_median'] = df[col].rolling(k).median()
    df['rolling_std'] = df[col].rolling(k).std()
    df['num_sigma'] = abs(df[col]-df['rolling_median'])/df['rolling_std']
    df[col] = np.where(df['num_sigma'] > threshold, df['rolling_median'], df[col])
    del df['rolling_median']
    del df['rolling_std']
    del df['num_sigma']
    return df


con = sqlite3.connect("C:/Users/Ben Bowes/PycharmProjects/hampt_rd_data (2).sqlite")

# Total hourly rainfall from gauge 036
rain_sql_036 = "SELECT * FROM datavalues WHERE SiteID=21 AND VariableID=1"
rain_df_036 = pd.read_sql_query(rain_sql_036, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_036.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_036.columns = ["MMPS-036"]
rain_df_036.sort_index(inplace=True)
rain_sum_036 = rain_df_036.resample("H").sum()

# Total hourly rainfall from gauge 039
rain_sql_039 = "SELECT * FROM datavalues WHERE SiteID=13 AND VariableID=1"
rain_df_039 = pd.read_sql_query(rain_sql_039, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_039.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_039.columns = ["MMPS-039"]
rain_df_039.sort_index(inplace=True)
rain_sum_039 = rain_df_039.resample("H").sum()

# Total hourly rainfall from gauge 140
rain_sql_140 = "SELECT * FROM datavalues WHERE SiteID=15 AND VariableID=1"
rain_df_140 = pd.read_sql_query(rain_sql_140, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_140.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_140.columns = ["MMPS-140"]
rain_df_140.sort_index(inplace=True)
rain_sum_140 = rain_df_140.resample("H").sum()

# Total hourly rainfall from gauge 149
rain_sql_149 = "SELECT * FROM datavalues WHERE SiteID=7 AND VariableID=1"
rain_df_149 = pd.read_sql_query(rain_sql_149, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_149.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_149.columns = ["MMPS-149"]
rain_df_149.sort_index(inplace=True)
rain_sum_149 = rain_df_149.resample("H").sum()

# Total hourly rainfall from gauge 163
rain_sql_163 = "SELECT * FROM datavalues WHERE SiteID=16 AND VariableID=1"
rain_df_163 = pd.read_sql_query(rain_sql_163, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_163.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_163.columns = ["MMPS-163"]
rain_df_163.sort_index(inplace=True)
rain_sum_163 = rain_df_163.resample("H").sum()

# Total hourly rainfall from gauge 175
rain_sql_175 = "SELECT * FROM datavalues WHERE SiteID=11 AND VariableID=1"
rain_df_175 = pd.read_sql_query(rain_sql_175, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_175.drop(["ValueID", "VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_175.columns = ["MMPS-175"]
rain_df_175.sort_index(inplace=True)
rain_sum_175 = rain_df_175.resample("H").sum()

# Total hourly rainfall from gauge 177
rain_sql_177 = "SELECT * FROM datavalues WHERE SiteID=12 AND VariableID=1"
rain_df_177 = pd.read_sql_query(rain_sql_177, con, index_col="Datetime", parse_dates=['Datetime'])
rain_df_177.drop(["ValueID","VariableID", "QCID", "SiteID"], axis=1, inplace=True)
rain_df_177.columns = ["MMPS-177"]
rain_df_177.sort_index(inplace=True)
rain_sum_177 = rain_df_177.resample("H").sum()

# # hourly tide for Sewell's Point
# tide_sql = "SELECT * FROM datavalues WHERE SiteID=17 AND VariableID=5"
# tide_df = pd.read_sql_query(tide_sql, con,index_col="Datetime", parse_dates=['Datetime'])
# tide_df.drop(["ValueID","VariableID","QCID","SiteID"], axis=1, inplace=True)
# tide_df.columns=["Tide"]
# tide_df.sort_index(inplace=True)
# tide_mean = tide_df.resample("H").mean()

# # Well MMPS-...
# gw_sql = "SELECT * FROM datavalues WHERE SiteID=9 AND VariableID=4"
# gw_df = pd.read_sql_query(gw_sql, con,index_col="Datetime", parse_dates=['Datetime'])
# gw_df.drop(["ValueID","VariableID","QCID","SiteID"], axis=1, inplace=True)
# gw_df.columns = ["GWL"]
# gw_df.sort_index(inplace=True)
# gw_hampel = hampel_filter(gw_df,"GWL",720)#Filtering for one day is 720 time steps
# gw_mean = gw_hampel.resample("H").mean()

# df = pd.concat([gw_mean,tide_mean,rain_sum_039,rain_sum_175])

# df = pd.merge(gw_mean,tide_mean,left_index=True,right_index=True)
# df = pd.merge(df, rain_sum_003, left_index=True, right_index=True)
df = pd.merge(rain_sum_036, rain_sum_039, left_index=True, right_index=True)
df = pd.merge(df, rain_sum_140, left_index=True, right_index=True)
df = pd.merge(df, rain_sum_149, left_index=True, right_index=True)
df = pd.merge(df, rain_sum_163, left_index=True, right_index=True)
df = pd.merge(df, rain_sum_175, left_index=True, right_index=True)
df = pd.merge(df, rain_sum_177, left_index=True, right_index=True)

df.to_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/HRSD_Rain_Data/all_gauges.csv")
