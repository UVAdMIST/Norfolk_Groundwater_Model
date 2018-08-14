"""
This script finds the start and end points of spikes in GWL for a single well's data.
The start and end points are used to create a new data frame of data that only includes the storm periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import argrelmax, find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from math import sqrt

# lag and forecast values
n_lag = 26
n_ahead = 19

# load dataset
data_path = "C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2018/"

dataset = pd.read_csv(data_path + "MMPS_043_no_blanks_SI.csv", index_col=None, parse_dates=True,
                      infer_datetime_format=True)

gwl_test = np.asarray(dataset["GWL"], dtype='float64')

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
filtered_start = []
filtered_peak = []
for i in range(0, len(potential_start), 1):  # potential start and peak list have same length
    if potential_start[i] != potential_start[i-1]:
        # print(potential_start[i], potential_peak[i])
        filtered_start.append(potential_start[i])
        filtered_peak.append(potential_peak[i])

# add lag data plus forecast to start and forecast data plus 24hrs to end
final_start = []
final_peak = []
for i, j in zip(filtered_start, filtered_peak):
    new_start = i - (n_lag + n_ahead)
    new_peak = j + n_ahead
    # print(new_start, new_peak)
    final_start.append(new_start)
    final_peak.append(new_peak)

# save dates of start and end lists
df_list = []
for i, j in zip(final_start, final_peak):
    print(i, j)
    df = dataset.iloc[i:j+1]
    df_list.append(df)

storms_df = pd.concat(df_list).drop_duplicates().reset_index(drop=True)

# plot observed gwl with start, end, and max f' points
fig, ax = plt.subplots()
ax.set_xlabel('Time Index')
ax.set_ylabel('GWL (ft)')
ax.plot(gwl_test)
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

storms_df.to_csv(data_path + "MMPS_043_no_blanks_SI_storms.csv", index=False)
