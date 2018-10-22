"""
By Benjamin Bowes, 10-14-18

This script reads results of the original (unbootstrapped) Rivanna model results for each well in Norfolk
and plots the months of 9-16, 1-17, and 5-18 (which are the months used with forecast data).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

# indicate which well to use
# well_list = ["043", "125", "129", "153", "155", "170", "175"]
well_list = ["175"]

base = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow"

for well in well_list:  # loop through all wells
    # specify output folder and load data
    out_folder = os.path.join(base, "rnn_lstm_comparison_results_fixed/mmps" + well)

    # load results from observed data
    rnn_full_obs = pd.read_csv(os.path.join(base, "Rivanna_results_shifted", "mmps" + well +
                                            "_results_full_bootstrap_rnn", "bs0_all_data_df.csv"),
                               index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    lstm_full_obs = pd.read_csv(os.path.join(base, "Rivanna_results_shifted", "mmps" + well +
                                             "_results_full_bootstrap_lstm", "bs0_all_data_df.csv"),
                                index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    rnn_storms_obs = pd.read_csv(os.path.join(base, "Rivanna_results_shifted", "mmps" + well +
                                              "_results_storm_bootstrap_rnn", "bs0_all_data_df.csv"),
                                 index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    lstm_storms_obs = pd.read_csv(os.path.join(base, "Rivanna_results_shifted", "mmps" + well +
                                               "_results_storm_bootstrap_lstm", "bs0_all_data_df.csv"),
                                  index_col="Datetime", parse_dates=True, infer_datetime_format=True)

    # load results from forecast data
    rnn_full_fcst = pd.read_csv(os.path.join(base, "Rivanna_results_fcst_shifted", "mmps" + well +
                                             "_results_full_bootstrap_fcst_rnn", "bs0_all_fcst_data_df.csv"),
                                index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    lstm_full_fcst = pd.read_csv(os.path.join(base, "Rivanna_results_fcst_shifted", "mmps" + well +
                                              "_results_full_bootstrap_fcst_lstm", "bs0_all_fcst_data_df.csv"),
                                 index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    rnn_storms_fcst = pd.read_csv(os.path.join(base, "Rivanna_results_fcst_shifted", "mmps" + well +
                                               "_results_storm_bootstrap_fcst_rnn", "bs0_all_fcst_data_df.csv"),
                                  index_col="Datetime", parse_dates=True, infer_datetime_format=True)
    lstm_storms_fcst = pd.read_csv(os.path.join(base, "Rivanna_results_fcst_shifted", "mmps" + well +
                                                "_results_storm_bootstrap_fcst_lstm", "bs0_all_fcst_data_df.csv"),
                                   index_col="Datetime", parse_dates=True, infer_datetime_format=True)

    # # make comparison plots for observed data
    # fig, axs = plt.subplots(3, 3, sharey=True, sharex=False, figsize=(8, 6))
    # secondary_ax = []
    # for plot_num, ax in enumerate(fig.axes):
    #     print(plot_num)
    #     if plot_num in [0, 1, 2]:
    #         forecast = "Pred. GWL t+1"
    #         row = 0
    #     if plot_num in [3, 4, 5]:
    #         forecast = "Pred. GWL t+9"
    #         row = 1
    #     if plot_num in [6, 7, 8]:
    #         forecast = "Pred. GWL t+18"
    #         row = 2
    #
    #     # create dfs of correct periods
    #     if plot_num in [0, 3, 6]:
    #         start, stop = "2016-08-31 21:00:00", "2016-10-01 13:00:00"
    #         col = 0
    #     if plot_num in [1, 4, 7]:
    #         start, stop = "2016-12-30 21:00:00", "2017-02-01 13:00:00"
    #         col = 1
    #     if plot_num in [2, 5, 8]:
    #         start, stop = "2018-04-30 21:00:00", "2018-05-31 08:00:00"
    #         col = 2
    #
    #     if plot_num== 0:
    #         y1_label = "t+1"
    #     if plot_num== 3:
    #         y1_label = "t+9"
    #     if plot_num== 6:
    #         y1_label = "t+18"
    #     if plot_num in [1, 2, 4, 5, 7, 8]:
    #         y1_label = ""
    #
    #     rnn_full_period = rnn_full_obs.loc[start:stop].reset_index()
    #     lstm_full_period = lstm_full_obs.loc[start:stop].reset_index()
    #     rnn_storms_period = rnn_storms_obs.loc[start:stop].reset_index()
    #     lstm_storms_period = lstm_storms_obs.loc[start:stop].reset_index()
    #
    #     ax = axs[row, col]
    #
    #     rnn_full_period[["Tide", "GWL", forecast]].plot(ax=ax, color=["k"], style=[":", '-', '-.'],
    #                                                     label=["Tide", "Obs. GWL", "Pred. GWL Full RNN"], legend=None)
    #     lstm_full_period[[forecast]].plot(ax=ax, color="b", style='-.', label="Pred. GWL Full LSTM", legend=None)
    #     # rnn_storms_period[[forecast]].plot(ax=ax, color="r", style='-.', label="Pred. GWL Storm RNN", legend=None)
    #     # lstm_storms_period[[forecast]].plot(ax=ax, color="g", style='-.', label="Pred. GWL Storm LSTM", legend=None)
    #
    #     begin, end = ax.get_xlim()
    #     ticks = np.arange(0, end, 24)  # (start,stop,increment)
    #     ax2 = ax.twinx()
    #     # ax2.set_ylim(ymax=1, ymin=0)
    #     # ax.set_ylim(ymax=4.5, ymin=-1.5)
    #     ax2.invert_yaxis()
    #     rnn_full_period["Precip."].plot.bar(ax=ax2, color="k")
    #     ax2.set_xticks([])
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(rnn_full_period.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
    #     ax.set_ylabel(y1_label)
    #     # ax.set_ylim(ymax=13,ymin=-3)
    #     # ax2.set_ylabel(y2_label)
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # # ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center
    # ax.legend(lines + lines2, ("Tide", "Obs. GWL", "Pred. GWL Full RNN", "Pred. GWL Full LSTM", "Precip."))
    # plt.tight_layout()
    # fig.text(.01, .55, "Hourly Avg GW/Tide Level (m)", horizontalalignment='center', verticalalignment='center',
    #          transform=ax.transAxes, rotation=90)
    # fig.text(.99, .55, "Total Hourly Precip. (mm)", horizontalalignment='center', verticalalignment='center',
    #          transform=ax.transAxes, rotation=-90)
    # # plt.subplots_adjust(left=0.07)
    # plt.show()
    #
    # # make comparison plots for forecast data
    # fig, axs = plt.subplots(3, 3, sharey=True, sharex=False, figsize=(8, 6))
    # secondary_ax = []
    # for plot_num, ax in enumerate(fig.axes):
    #     print(plot_num)
    #     if plot_num in [0, 1, 2]:
    #         observed = "Obs. GWL t+1"
    #         tide = "tide(t+1)"
    #         rain = "rain(t+1)"
    #         forecast = "Pred. GWL t+1"
    #         row = 0
    #     if plot_num in [3, 4, 5]:
    #         observed = "Obs. GWL t+9"
    #         tide = "tide(t+9)"
    #         rain = "rain(t+9)"
    #         forecast = "Pred. GWL t+9"
    #         row = 1
    #     if plot_num in [6, 7, 8]:
    #         observed = "Obs. GWL t+18"
    #         tide = "tide(t+18)"
    #         rain = "rain(t+18)"
    #         forecast = "Pred. GWL t+18"
    #         row = 2
    #
    #     # create dfs of correct periods
    #     if plot_num in [0, 3, 6]:
    #         start, stop = "2016-08-31 21:00:00", "2016-10-01 13:00:00"
    #         col = 0
    #     if plot_num in [1, 4, 7]:
    #         start, stop = "2016-12-30 21:00:00", "2017-02-01 13:00:00"
    #         col = 1
    #     if plot_num in [2, 5, 8]:
    #         start, stop = "2018-04-30 21:00:00", "2018-05-31 23:00:00"
    #         col = 2
    #
    #     if plot_num == 0:
    #         y1_label = "t+1"
    #     if plot_num == 3:
    #         y1_label = "t+9"
    #     if plot_num == 6:
    #         y1_label = "t+18"
    #     if plot_num in [1, 2, 4, 5, 7, 8]:
    #         y1_label = ""
    #
    #     rnn_full_period = rnn_full_fcst.loc[start:stop].reset_index()
    #     lstm_full_period = lstm_full_fcst.loc[start:stop].reset_index()
    #     rnn_storms_period = rnn_storms_fcst.loc[start:stop].reset_index()
    #     lstm_storms_period = lstm_storms_fcst.loc[start:stop].reset_index()
    #
    #     ax = axs[row, col]
    #
    #     rnn_full_period[[tide, observed, forecast]].plot(ax=ax, color=["k"], style=[":", '-', '-.'],
    #                                                     label=["Tide", "Obs. GWL", "Pred. GWL Full RNN"], legend=None)
    #     lstm_full_period[[forecast]].plot(ax=ax, color="b", style='-.', label="Pred. GWL Full LSTM", legend=None)
    #     rnn_storms_period[[forecast]].plot(ax=ax, color="r", style='-.', label="Pred. GWL Storm RNN", legend=None)
    #     lstm_storms_period[[forecast]].plot(ax=ax, color="g", style='-.', label="Pred. GWL Storm LSTM", legend=None)
    #
    #     begin, end = ax.get_xlim()
    #     ticks = np.arange(0, end, 24)  # (start,stop,increment)
    #     ax2 = ax.twinx()
    #     # ax2.set_ylim(ymax=1, ymin=0)
    #     # ax.set_ylim(ymax=4.5, ymin=-1.5)
    #     ax2.invert_yaxis()
    #     rnn_full_period[rain].plot.bar(ax=ax2, color="k")
    #     ax2.set_xticks([])
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(rnn_full_period.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
    #     ax.set_ylabel(y1_label)
    #     # ax.set_ylim(ymax=13,ymin=-3)
    #     # ax2.set_ylabel(y2_label)
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # # ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center
    # ax.legend(lines + lines2, ("Tide", "Obs. GWL", "Pred. GWL Full RNN", "Pred. GWL Full LSTM",
    #                            "Pred. GWL Storm RNN", "Pred. GWL Storm LSTM", "Precip."))
    # plt.tight_layout()
    # fig.text(.01, .55, "Hourly Avg GW/Tide Level (m)", horizontalalignment='center', verticalalignment='center',
    #          transform=ax.transAxes, rotation=90)
    # fig.text(.99, .55, "Total Hourly Precip. (mm)", horizontalalignment='center', verticalalignment='center',
    #          transform=ax.transAxes, rotation=-90)
    # # plt.subplots_adjust(left=0.07)
    # plt.show()

    # individual plots for each test period
    starts = ["2016-08-31 21:00:00", "2016-12-30 21:00:00", "2018-04-30 21:00:00"]
    stops = ["2016-10-01 13:00:00", "2017-02-01 13:00:00", "2018-05-31 08:00:00"]
    for start, stop in zip(starts, stops):
        print(start, stop)
        rnn_full_period = rnn_full_obs.loc[start:stop].reset_index()
        lstm_full_period = lstm_full_obs.loc[start:stop].reset_index()
        rnn_fcst_period = rnn_full_fcst.loc[start:stop].reset_index()
        lstm_fcst_period = lstm_full_fcst.loc[start:stop].reset_index()

        fig, axs = plt.subplots(3, sharey=False, sharex=True, figsize=(6, 8))
        for plot_num, ax in enumerate(fig.axes):
            print(plot_num)
            if plot_num == 0:
                forecast = "Pred. GWL t+1"
                y1_label = ""
                y2_label = ""
            if plot_num == 1:
                forecast = "Pred. GWL t+9"
                y1_label = "Hourly Avg GW/Tide Level (m)"
                y2_label = "Total Hourly Precip. (mm)"
            if plot_num == 2:
                forecast = "Pred. GWL t+18"
                y1_label = ""
                y2_label = ""
            print(forecast)

            ax = axs[plot_num]

            rnn_full_period[["Tide", "GWL", forecast]].plot(ax=ax, color=["k", 'k', 'c'], style=[":", '-', '-.'], legend=None)
            lstm_full_period[[forecast]].plot(ax=ax, color="b", style='-.', legend=None)
            rnn_fcst_period[[forecast]].plot(ax=ax, color="r", style='-.', legend=None)
            lstm_fcst_period[[forecast]].plot(ax=ax, color="g", style='-.', legend=None)

            begin, end = ax.get_xlim()
            ticks = np.arange(0, end, 24)  # (start,stop,increment)
            ax2 = ax.twinx()
            # ax2.set_ylim(ymax=1, ymin=0)
            # ax.set_ylim(ymax=4.5, ymin=-1.5)
            ax2.invert_yaxis()
            rnn_full_period["Precip."].plot.bar(ax=ax2, color="k", legend=None)
            ax2.set_xticks([])
            ax.set_xticks(ticks)
            ax.set_xticklabels(rnn_full_period.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
            ax.set_ylabel(y1_label)
            # ax.set_ylim(ymax=13,ymin=-3)
            ax2.set_ylabel(y2_label)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
            # ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center
        ax.legend(lines + lines2, ("Obs. Tide", "Obs. GWL", "Full RNN", "Full LSTM", "Full RNN Fcst.",
                                   "Full LSTM Fcst.", "Obs. Precip."), bbox_to_anchor=(0.78, -0.38), ncol=2)
        plt.tight_layout()
        plt.show()
        # plot_path = path + "%s_t%s.pdf" % (storm.name, i)
        # plt.savefig(plot_path, dpi=300)
        plt.close()
