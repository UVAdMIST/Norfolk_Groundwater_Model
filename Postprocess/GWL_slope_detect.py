import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import argrelmax, find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from math import sqrt

path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_results_18hr_rnn/"
lstm_path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_results_18hr/"

# load dataset from model results
data_cols = ["Datetime", "GWL", "Tide", "Precip.Avg", "Pred. GWL t+1", "Pred. GWL t+9", "Pred. GWL t+18"]

all_data_df = pd.read_csv(path + "all_data_df.csv", usecols=data_cols, index_col=None, parse_dates=True,
                          infer_datetime_format=True)

lstm_data = pd.read_csv(lstm_path + "all_data_df.csv", usecols=data_cols, index_col=None, parse_dates=True,
                        infer_datetime_format=True)

gwl_test = np.asarray(all_data_df["GWL"], dtype='float64')

# use first and second derivative
fig = plt.figure(figsize=(8, 14))
gs = gridspec.GridSpec(7, 1)

ax0 = plt.subplot(gs[0])
ax0.set_title('GWL')
ax0.plot(gwl_test)

first_dev = np.gradient(gwl_test)
ax1 = plt.subplot(gs[1])
ax1.set_title('1st derivative')
ax1.plot(first_dev)

second_dev = np.gradient(first_dev)
ax2 = plt.subplot(gs[2])
ax2.set_title('2nd derivative')
ax2.plot(second_dev)

first_dev_clipped = np.clip(np.abs(np.gradient(first_dev)), 0.0001, 2)
ax3 = plt.subplot(gs[3])
ax3.set_title('first derivative absolute value + clipping')
ax3.plot(first_dev_clipped)

second_dev_clipped = np.clip(np.abs(np.gradient(second_dev)), 0.0001, 2)
ax4 = plt.subplot(gs[4])
ax4.set_title('second derivative absolute value + clipping')
ax4.plot(second_dev_clipped)

first_dev_smoothed = gaussian_filter1d(first_dev, 5)
ax5 = plt.subplot(gs[5])
ax5.set_title('Smoothing applied to 1st derivative')
ax5.plot(first_dev_smoothed)

second_dev_smoothed = gaussian_filter1d(second_dev_clipped, 5)
ax6 = plt.subplot(gs[6])
ax6.set_title('Smoothing applied to second derivative')
ax6.plot(second_dev_smoothed)

plt.tight_layout()
plt.show()

# find peaks
# max_idx = argrelmax(second_dev_smoothed)[0]
found_peaks = find_peaks(gwl_test, prominence=0.05, distance=50)
# max_1stdev = argrelmax(first_dev_smoothed)[0]
# print(max_idx, max_1stdev)

# find indices where first derivative == 0
first_dev_zeros = np.where(first_dev == 0)

# find location of max first derivative
found_first_dev = find_peaks(first_dev_smoothed, height=0.001)

# found_second_dev = find_peaks(second_dev_smoothed, height=0.00015, distance=100, prominence=0.0002)
# found_second_dev = np.asarray(found_second_dev[0])

# find indices of zero values that bracket max value
start_list = []
end_list = []
for i in found_first_dev[0]:
    min_list = []
    max_list = []
    for j in first_dev_zeros[0]:
        if i < j:
            min_list.append(j)
        if i > j:
            max_list.append(j)
        else:
            continue
    end_list.append(min(min_list))
    start_list.append(max(max_list))

# compare start values to see if there is a peak between them, if not select the one with the lowest gwl value
# first, get gwl values for start values
# start_gwl = []
# for i in start_list:
#     gwl = gwl_test[i]
#     start_gwl.append(gwl)
#
# new_start = []
# for i in range(0, len(start_list), 1):
#     for j in range(0, len(found_peaks[0]), 1):
#         if i < len(start_list)-1:
#             if start_list[i] < found_peaks[0][j]:
#                 if start_list[i+1] < found_peaks[0][j]:
#                     print(start_list[i])

# compare peak values to see if there is a start between them, if not select the one with the highest gwl value


# create pairs of start and peak values
potential_start = []
potential_peak = []
for i in range(0, len(start_list), 1):
    for j in range(0, len(found_peaks[0]), 1):
        if i < len(start_list)-1:
            if start_list[i] < found_peaks[0][j]:
                if found_peaks[0][j] < start_list[i+1]:
                    # print(start_list[i], found_peaks[0][j])
                    potential_start.append(start_list[i])
                    potential_peak.append(found_peaks[0][j])
        if i == len(start_list)-1:
            if start_list[i] < found_peaks[0][j]:
                # print(start_list[i], found_peaks[0][j])
                potential_start.append(start_list[i])
                potential_peak.append(found_peaks[0][j])

# filter pairs of values to remove ones that have the same start
final_start = []
final_peak = []
for i in range(0, len(potential_start), 1):  # potential start and peak list have same length
    if potential_start[i] != potential_start[i-1]:
        # print(potential_start[i], potential_peak[i])
        final_start.append(potential_start[i])
        final_peak.append(potential_peak[i])

# save dates of start and end lists
df_list = []
for i, j in zip(final_start, final_peak):
    print(i, j)
    df = all_data_df.iloc[i:j+1]
    df_list.append(df)

storms_df = pd.concat(df_list)
storms_df.to_csv(path + "rising_limb_df.csv")

# plot observed gwl with start, end, and max f' points
fig, ax = plt.subplots()
ax.set_xlabel('Time Index')
ax.set_ylabel('GWL (ft)')
ax.plot(gwl_test)
ax.plot(all_data_df[["Pred. GWL t+18"]], ":", label='RNN t+18 forecast')
ax.plot(lstm_data[["Pred. GWL t+18"]], "--", label='LSTM t+18 forecast')
# ax.scatter(found_first_dev[0], gwl_test[found_first_dev[0]], marker='o', color='blue', label="Max f'")
ax.scatter(final_start, gwl_test[final_start], marker='x', color='red', label='Start')
ax.scatter(final_peak, gwl_test[final_peak], marker='P', color='k', label='Peak')
# ax.scatter(start_list, gwl_test[start_list], marker='x', color='red', label='Start')
# ax.scatter(end_list, gwl_test[end_list], marker='^', color='green', label='End')
# ax.scatter(found_peaks[0], gwl_test[found_peaks[0]], marker='P', color='k', label='Peak')
# ax.scatter(found_second_dev, gwl_test[found_second_dev], marker='*', color='purple', label='second_dev')
# ax.scatter(max_idx, gwl_test[max_idx], marker='p', color='orange', label='max_idx')
# ax.scatter(found_peaks[0], gwl_test[found_peaks[0]], marker='P', color='k', label='found_peaks')
plt.legend()
plt.tight_layout()
plt.show()

# plot storm periods
# fig, ax = plt.subplots()
# ax.set_xlabel('Time Index')
# ax.set_ylabel('GWL (ft)')
# ax.plot(storms_df[["GWL"]], label="Obs.")
# ax.plot(storms_df[["Pred. GWL t+1"]], label="Forecast")
# plt.legend()
# plt.tight_layout()
# plt.show()

forecast_list = ["Pred. GWL t+1", "Pred. GWL t+9", "Pred. GWL t+18"]
RMSE_forecast = []
for i in forecast_list:
    mse = mean_squared_error(storms_df[["GWL"]], storms_df[[i]])
    rmse = sqrt(mse)
    RMSE_forecast.append(rmse)

NSE_forecast = []
for i in forecast_list:
    num_diff = np.subtract(np.asarray(storms_df[["GWL"]]), np.asarray(storms_df[[i]]))
    num_sq = np.square(num_diff)
    numerator = sum(num_sq)
    denom_diff = np.subtract(np.asarray(storms_df[["GWL"]]), np.mean(np.asarray(storms_df[["GWL"]])))
    denom_sq = np.square(denom_diff)
    denominator = sum(denom_sq)
    if denominator == 0:
        nse = 'NaN'
    else:
        nse = 1-(numerator/denominator)
    NSE_forecast.append(nse[0])

storms_metrics_df = pd.DataFrame([RMSE_forecast, NSE_forecast])
storms_metrics_df.columns = forecast_list
storms_metrics_df = storms_metrics_df.transpose()
storms_metrics_df.columns = ["RMSE", "NSE"]
storms_metrics_df.to_csv(path + "rising_limb_metrics.csv")
