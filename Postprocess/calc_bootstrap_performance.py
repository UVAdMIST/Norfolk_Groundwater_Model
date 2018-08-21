"""
This script reads all the bootstrap performance result files, plots histograms, and calculates averages.
t-tests are done to compute p-values and confidence intervals are computed
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams.update({'font.size': 8})

# specify folder locations
rnn_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_results_storm_bootstrap_rnn/"
lstm_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/mmps043_results_storm_bootstrap_lstm/"

# extract forecast data for RNN
rnn_rmse_t1_list, rnn_rmse_t9_list, rnn_rmse_t18_list = [], [], []
rnn_nse_t1_list, rnn_nse_t9_list, rnn_nse_t18_list = [], [], []
rnn_mae_t1_list, rnn_mae_t9_list, rnn_mae_t18_list = [], [], []

for file in os.listdir(rnn_results_folder):
    data = rnn_results_folder + file
    if file.endswith("_RMSE.csv"):
        # print(file)
        rmse_df = pd.read_csv(data)
        rmse_t1, rmse_t9, rmse_t18 = rmse_df[["0"]].iloc[0], rmse_df[["0"]].iloc[8], rmse_df[["0"]].iloc[17]
        rnn_rmse_t1_list.append(rmse_t1[0])
        rnn_rmse_t9_list.append(rmse_t9[0])
        rnn_rmse_t18_list.append(rmse_t18[0])
    if file.endswith("_NSE.csv"):
        nse_df = pd.read_csv(data)
        nse_t1, nse_t9, nse_t18 = nse_df[["0"]].iloc[0], nse_df[["0"]].iloc[8], nse_df[["0"]].iloc[17]
        rnn_nse_t1_list.append(nse_t1[0])
        rnn_nse_t9_list.append(nse_t9[0])
        rnn_nse_t18_list.append(nse_t18[0])
    if file.endswith("_MAE.csv"):
        mae_df = pd.read_csv(data)
        mae_t1, mae_t9, mae_t18 = mae_df[["0"]].iloc[0], mae_df[["0"]].iloc[8], mae_df[["0"]].iloc[17]
        rnn_mae_t1_list.append(mae_t1[0])
        rnn_mae_t9_list.append(mae_t9[0])
        rnn_mae_t18_list.append(mae_t18[0])

# write extracted data to data frames
rnn_RMSE_df = pd.DataFrame([rnn_rmse_t1_list, rnn_rmse_t9_list, rnn_rmse_t18_list]).transpose()
rnn_RMSE_df.columns = ["t+1", "t+9", "t+18"]
rnn_NSE_df = pd.DataFrame([rnn_nse_t1_list, rnn_nse_t9_list, rnn_nse_t18_list]).transpose()
rnn_NSE_df.columns = ["t+1", "t+9", "t+18"]
rnn_MAE_df = pd.DataFrame([rnn_mae_t1_list, rnn_mae_t9_list, rnn_mae_t18_list]).transpose()
rnn_MAE_df.columns = ["t+1", "t+9", "t+18"]

# extract forecast data for LSTM
lstm_rmse_t1_list, lstm_rmse_t9_list, lstm_rmse_t18_list = [], [], []
lstm_nse_t1_list, lstm_nse_t9_list, lstm_nse_t18_list = [], [], []
lstm_mae_t1_list, lstm_mae_t9_list, lstm_mae_t18_list = [], [], []

for file in os.listdir(lstm_results_folder):
    data = lstm_results_folder + file
    if file.endswith("_RMSE.csv"):
        # print(file)
        rmse_df = pd.read_csv(data)
        rmse_t1, rmse_t9, rmse_t18 = rmse_df[["0"]].iloc[0], rmse_df[["0"]].iloc[8], rmse_df[["0"]].iloc[17]
        lstm_rmse_t1_list.append(rmse_t1[0])
        lstm_rmse_t9_list.append(rmse_t9[0])
        lstm_rmse_t18_list.append(rmse_t18[0])
    if file.endswith("_NSE.csv"):
        nse_df = pd.read_csv(data)
        nse_t1, nse_t9, nse_t18 = nse_df[["0"]].iloc[0], nse_df[["0"]].iloc[8], nse_df[["0"]].iloc[17]
        lstm_nse_t1_list.append(nse_t1[0])
        lstm_nse_t9_list.append(nse_t9[0])
        lstm_nse_t18_list.append(nse_t18[0])
    if file.endswith("_MAE.csv"):
        mae_df = pd.read_csv(data)
        mae_t1, mae_t9, mae_t18 = mae_df[["0"]].iloc[0], mae_df[["0"]].iloc[8], mae_df[["0"]].iloc[17]
        lstm_mae_t1_list.append(mae_t1[0])
        lstm_mae_t9_list.append(mae_t9[0])
        lstm_mae_t18_list.append(mae_t18[0])

# write extracted data to data frames
lstm_RMSE_df = pd.DataFrame([lstm_rmse_t1_list, lstm_rmse_t9_list, lstm_rmse_t18_list]).transpose()
lstm_RMSE_df.columns = ["t+1", "t+9", "t+18"]
lstm_NSE_df = pd.DataFrame([lstm_nse_t1_list, lstm_nse_t9_list, lstm_nse_t18_list]).transpose()
lstm_NSE_df.columns = ["t+1", "t+9", "t+18"]
lstm_MAE_df = pd.DataFrame([lstm_mae_t1_list, lstm_mae_t9_list, lstm_mae_t18_list]).transpose()
lstm_MAE_df.columns = ["t+1", "t+9", "t+18"]

# plot histograms
plt.figure(1, figsize=(6, 6))

ax1 = plt.subplot(321)
rnn_RMSE_df.hist(ax=ax1, column="t+1", bins=10)
ax1.set_title("")
ax1.set_ylabel("t+1")

ax2 = plt.subplot(323)
rnn_RMSE_df.hist(ax=ax2, column="t+9", bins=10)
ax2.set_ylabel("Count")
ax2.set_title("")
ax2.set_ylabel("t+9")

ax3 = plt.subplot(325)
rnn_RMSE_df.hist(ax=ax3, column="t+18", bins=10)
ax3.set_title("")
ax3.set_ylabel("t+18")

ax4 = plt.subplot(322)
lstm_RMSE_df.hist(ax=ax4, column="t+1", bins=10)
ax4.set_title("")

ax5 = plt.subplot(324)
lstm_RMSE_df.hist(ax=ax5, column="t+9", bins=10)
ax5.set_title("")

ax6 = plt.subplot(326)
lstm_RMSE_df.hist(ax=ax6, column="t+18", bins=10)
ax6.set_title("")

plt.gcf().text(0.5, 0.05, "RMSE (m)")

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()
# plt.savefig("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results/full_testset_metrics.png", dpi=300)

# perform t-tests
rmse_t1_tvalue, rmse_t1_pvalue = stats.ttest_ind(rnn_rmse_t1_list, lstm_rmse_t1_list)
rmse_t9_tvalue, rmse_t9_pvalue = stats.ttest_ind(rnn_rmse_t9_list, lstm_rmse_t9_list)
rmse_t18_tvalue, rmse_t18_pvalue = stats.ttest_ind(rnn_rmse_t18_list, lstm_rmse_t18_list)

nse_t1_tvalue, nse_t1_pvalue = stats.ttest_ind(rnn_nse_t1_list, lstm_nse_t1_list)
nse_t9_tvalue, nse_t9_pvalue = stats.ttest_ind(rnn_nse_t9_list, lstm_nse_t9_list)
nse_t18_tvalue, nse_t18_pvalue = stats.ttest_ind(rnn_nse_t18_list, lstm_nse_t18_list)

mae_t1_tvalue, mae_t1_pvalue = stats.ttest_ind(rnn_mae_t1_list, lstm_mae_t1_list)
mae_t9_tvalue, mae_t9_pvalue = stats.ttest_ind(rnn_mae_t9_list, lstm_mae_t9_list)
mae_t18_tvalue, mae_t18_pvalue = stats.ttest_ind(rnn_mae_t18_list, lstm_mae_t18_list)

# calculate means
rmse_t1_mean = rnn_RMSE_df[["t+1"]].mean()[0]
lstm_t1_mean = lstm_RMSE_df[["t+1"]].mean()[0]
rmse_t9_mean = rnn_RMSE_df[["t+9"]].mean()[0]
lstm_t9_mean = lstm_RMSE_df[["t+9"]].mean()[0]
rmse_t18_mean = rnn_RMSE_df[["t+18"]].mean()[0]
lstm_t18_mean = lstm_RMSE_df[["t+18"]].mean()[0]
