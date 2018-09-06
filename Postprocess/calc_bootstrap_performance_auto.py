"""
This script reads all the bootstrap performance result files, plots histograms, and calculates averages.
t-tests are done to compute p-values and confidence intervals are computed
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams.update({'font.size': 8})

# well_list = ["043", "125", "129", "153", "155", "170", "175"]
well_list = ["125"]

for well in well_list:  # loop through all wells
    # specify folder locations
    out_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results/mmps" + well

    rnn_full_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/Rivanna_results/mmps" + well +\
                              "_results_full_bootstrap_rnn/"
    lstm_full_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/Rivanna_results/mmps" + well +\
                               "_results_full_bootstrap_lstm/"
    rnn_storms_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/Rivanna_results/mmps" + well +\
                                "_results_storm_bootstrap_rnn/"
    lstm_storms_results_folder = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/Rivanna_results/mmps" + well +\
                                 "_results_storm_bootstrap_lstm/"

    folder_list = [rnn_full_results_folder, lstm_full_results_folder, rnn_storms_results_folder,
                   lstm_storms_results_folder]

    rmse_df_list = []
    nse_df_list = []
    mae_df_list = []
    rmse_storms_df_list = []
    nse_storms_df_list = []
    mae_storms_df_list = []

    for folder in folder_list:
        folder_name1 = folder.split("/")[6].split("_")[2]
        folder_name2 = folder.split("/")[6].split("_")[4]
        folder_name = folder_name1 + "_" + folder_name2
        print(folder_name)

        rmse_t1_list, rmse_t9_list, rmse_t18_list = [], [], []
        nse_t1_list, nse_t9_list, nse_t18_list = [], [], []
        mae_t1_list, mae_t9_list, mae_t18_list = [], [], []

        rmse_storms_t1_list, rmse_storms_t9_list, rmse_storms_t18_list = [], [], []
        nse_storms_t1_list, nse_storms_t9_list, nse_storms_t18_list = [], [], []
        mae_storms_t1_list, mae_storms_t9_list, mae_storms_t18_list = [], [], []

        count = 0
        for file in os.listdir(folder):  # extract forecast data
            if count % 100 == 0:
                print(folder, "count is", count)
            data = folder + file
            if file.endswith("_RMSE.csv"):
                # print(file)
                rmse_df = pd.read_csv(data)
                rmse_t1, rmse_t9, rmse_t18 = rmse_df[["0"]].iloc[0], rmse_df[["0"]].iloc[8], rmse_df[["0"]].iloc[17]
                rmse_t1_list.append(rmse_t1[0])
                rmse_t9_list.append(rmse_t9[0])
                rmse_t18_list.append(rmse_t18[0])
            if file.endswith("_NSE.csv"):
                nse_df = pd.read_csv(data)
                nse_t1, nse_t9, nse_t18 = nse_df[["0"]].iloc[0], nse_df[["0"]].iloc[8], nse_df[["0"]].iloc[17]
                nse_t1_list.append(nse_t1[0])
                nse_t9_list.append(nse_t9[0])
                nse_t18_list.append(nse_t18[0])
            if file.endswith("_MAE.csv"):
                mae_df = pd.read_csv(data)
                mae_t1, mae_t9, mae_t18 = mae_df[["0"]].iloc[0], mae_df[["0"]].iloc[8], mae_df[["0"]].iloc[17]
                mae_t1_list.append(mae_t1[0])
                mae_t9_list.append(mae_t9[0])
                mae_t18_list.append(mae_t18[0])
            if file.endswith("_RMSE_storms.csv"):
                # print(file)
                rmse_df = pd.read_csv(data)
                rmse_storms_t1, rmse_storms_t9, rmse_storms_t18 = rmse_df[["0"]].iloc[0], rmse_df[["0"]].iloc[1],\
                                                                  rmse_df[["0"]].iloc[2]
                rmse_storms_t1_list.append(rmse_storms_t1[0])
                rmse_storms_t9_list.append(rmse_storms_t9[0])
                rmse_storms_t18_list.append(rmse_storms_t18[0])
            if file.endswith("_NSE_storms.csv"):
                nse_df = pd.read_csv(data)
                nse_storms_t1, nse_storms_t9, nse_storms_t18 = nse_df[["0"]].iloc[0], nse_df[["0"]].iloc[1],\
                                                               nse_df[["0"]].iloc[2]
                nse_storms_t1_list.append(nse_storms_t1[0])
                nse_storms_t9_list.append(nse_storms_t9[0])
                nse_storms_t18_list.append(nse_storms_t18[0])
            if file.endswith("_MAE_storms.csv"):
                mae_df = pd.read_csv(data)
                mae_storms_t1, mae_storms_t9, mae_storms_t18 = mae_df[["0"]].iloc[0], mae_df[["0"]].iloc[1],\
                                                               mae_df[["0"]].iloc[2]
                mae_storms_t1_list.append(mae_storms_t1[0])
                mae_storms_t9_list.append(mae_storms_t9[0])
                mae_storms_t18_list.append(mae_storms_t18[0])
            count += 1
        # write extracted data to data frames
        folder_RMSE_df = pd.DataFrame([rmse_t1_list, rmse_t9_list, rmse_t18_list]).transpose()
        folder_RMSE_df.columns = [(folder_name + "_t+1"), (folder_name + "_t+9"), (folder_name + "_t+18")]
        # print("folder rmse df", folder_RMSE_df.head())
        folder_NSE_df = pd.DataFrame([nse_t1_list, nse_t9_list, nse_t18_list]).transpose()
        folder_NSE_df.columns = [(folder_name + "_t+1"), (folder_name + "_t+9"), (folder_name + "_t+18")]
        folder_MAE_df = pd.DataFrame([mae_t1_list, mae_t9_list, mae_t18_list]).transpose()
        folder_MAE_df.columns = [(folder_name + "_t+1"), (folder_name + "_t+9"), (folder_name + "_t+18")]
        if folder_name1 == "full":
            folder_storms_RMSE_df = pd.DataFrame([rmse_storms_t1_list, rmse_storms_t9_list, rmse_storms_t18_list])\
                .transpose()
            folder_storms_RMSE_df.columns = [(folder_name + "storms_t+1"), (folder_name + "storms_t+9"),
                                             (folder_name + "storms_t+18")]
            # print("folder rmse df", folder_RMSE_df.head())
            folder_storms_NSE_df = pd.DataFrame([nse_storms_t1_list, nse_storms_t9_list, nse_storms_t18_list])\
                .transpose()
            folder_storms_NSE_df.columns = [(folder_name + "storms_t+1"), (folder_name + "storms_t+9"),
                                            (folder_name + "storms_t+18")]
            folder_storms_MAE_df = pd.DataFrame([mae_storms_t1_list, mae_storms_t9_list, mae_storms_t18_list])\
                .transpose()
            folder_storms_MAE_df.columns = [(folder_name + "storms_t+1"), (folder_name + "storms_t+9"),
                                            (folder_name + "storms_t+18")]

        # append folder dataframes to lists
        rmse_df_list.append(folder_RMSE_df)
        nse_df_list.append(folder_NSE_df)
        mae_df_list.append(folder_MAE_df)
        if folder_name1 == "full":
            rmse_df_list.append(folder_storms_RMSE_df)
            nse_df_list.append(folder_storms_NSE_df)
            mae_df_list.append(folder_storms_MAE_df)

    # concat data to well dfs
    rmse_df = pd.concat(rmse_df_list, axis=1)
    rmse_df = rmse_df[:948]
    nse_df = pd.concat(nse_df_list, axis=1)
    nse_df = nse_df[:1000]
    mae_df = pd.concat(mae_df_list, axis=1)
    mae_df = mae_df[:1000]
    # rmse_storms_df = pd.concat(rmse_storms_df_list, axis=1)
    # nse_storms_df = pd.concat(nse_storms_df_list, axis=1)
    # mae_storms_df = pd.concat(mae_storms_df_list, axis=1)

    # save well dfs
    rmse_df.to_csv(os.path.join(out_folder, "rmse_df.csv"), index=False)
    nse_df.to_csv(os.path.join(out_folder, "nse_df.csv"), index=False)
    mae_df.to_csv(os.path.join(out_folder, "mae_df.csv"), index=False)
    # rmse_storms_df.to_csv(os.path.join(out_folder, "rmse_storms_df.csv"), index=False)
    # nse_storms_df.to_csv(os.path.join(out_folder, "nse_storms_df.csv"), index=False)
    # mae_storms_df.to_csv(os.path.join(out_folder, "mae_storms_df.csv"), index=False)

    # plot histograms of RMSE for individual well
    col_list = rmse_df.columns

    plt.figure(1, figsize=(6, 9))
    for i in range(0, len(col_list), 1):
        ax = plt.subplot(6, 3, i+1)
        rmse_df.hist(ax=ax, column=col_list[i], bins=15)
        bs_type = col_list[i].split("_")[0]
        model_type = col_list[i].split("_")[1]
        ax.set_title("")
        if i % 3 == 0:
            ax.set_ylabel(bs_type + " " + model_type)
        if i < 3:
            ax.set_title(col_list[i].split("_")[2])
    plt.tight_layout()
    plt.gcf().text(0.5, 0.05, "RMSE (m)")
    plt.subplots_adjust(bottom=0.1)
    # plt.show()
    plt.savefig(os.path.join(out_folder, "rmse_hists.png"), dpi=300)
    plt.close()

    # perform t-tests
    rnn_full_storm_tvalues, rnn_full_storm_pvalues = [], []
    lstm_full_storm_tvalues, lstm_full_storm_pvalues = [], []
    rnn_lstm_full_tvalues, rnn_lstm_full_pvalues = [], []
    rnn_lstm_storm_tvalues, rnn_lstm_storm_pvalues = [], []
    fullRNNStorms_vs_stormRNN_tvalues, fullRNNStorms_vs_stormRNN_pvalues = [], []
    fullLSTMStorms_vs_stormLSTM_tvalues, fullLSTMStorms_vs_stormLSTM_pvalues = [], []

    rnn_full_storm_t1_t, rnn_full_storm_t1_p = stats.ttest_ind(rmse_df["full_rnn_t+1"], rmse_df["storm_rnn_t+1"])
    rnn_full_storm_t9_t, rnn_full_storm_t9_p = stats.ttest_ind(rmse_df["full_rnn_t+9"], rmse_df["storm_rnn_t+9"])
    rnn_full_storm_t18_t, rnn_full_storm_t18_p = stats.ttest_ind(rmse_df["full_rnn_t+18"], rmse_df["storm_rnn_t+18"])
    rnn_full_storm_tvalues.append([rnn_full_storm_t1_t, rnn_full_storm_t9_t, rnn_full_storm_t18_t])
    rnn_full_storm_pvalues.append([rnn_full_storm_t1_p, rnn_full_storm_t9_p, rnn_full_storm_t18_p])

    lstm_full_storm_t1_t, lstm_full_storm_t1_p = stats.ttest_ind(rmse_df["full_lstm_t+1"], rmse_df["storm_lstm_t+1"])
    lstm_full_storm_t9_t, lstm_full_storm_t9_p = stats.ttest_ind(rmse_df["full_lstm_t+9"], rmse_df["storm_lstm_t+9"])
    lstm_full_storm_t18_t, lstm_full_storm_t18_p = stats.ttest_ind(rmse_df["full_lstm_t+18"],rmse_df["storm_lstm_t+18"])
    lstm_full_storm_tvalues.append([lstm_full_storm_t1_t, lstm_full_storm_t9_t, lstm_full_storm_t18_t])
    lstm_full_storm_pvalues.append([lstm_full_storm_t1_p, lstm_full_storm_t9_p, lstm_full_storm_t18_p])

    rnn_lstm_full_t1_t, rnn_lstm_full_t1_p = stats.ttest_ind(rmse_df["full_rnn_t+1"], rmse_df["full_lstm_t+1"])
    rnn_lstm_full_t9_t, rnn_lstm_full_t9_p = stats.ttest_ind(rmse_df["full_rnn_t+9"], rmse_df["full_lstm_t+9"])
    rnn_lstm_full_t18_t, rnn_lstm_full_t18_p = stats.ttest_ind(rmse_df["full_rnn_t+18"], rmse_df["full_lstm_t+18"])
    rnn_lstm_full_tvalues.append([rnn_lstm_full_t1_t, rnn_lstm_full_t9_t, rnn_lstm_full_t18_t])
    rnn_lstm_full_pvalues.append([rnn_lstm_full_t1_p, rnn_lstm_full_t9_p, rnn_lstm_full_t18_p])

    rnn_lstm_storm_t1_t, rnn_lstm_storm_t1_p = stats.ttest_ind(rmse_df["storm_rnn_t+1"], rmse_df["storm_lstm_t+1"])
    rnn_lstm_storm_t9_t, rnn_lstm_storm_t9_p = stats.ttest_ind(rmse_df["storm_rnn_t+9"], rmse_df["storm_lstm_t+9"])
    rnn_lstm_storm_t18_t, rnn_lstm_storm_t18_p = stats.ttest_ind(rmse_df["storm_rnn_t+18"], rmse_df["storm_lstm_t+18"])
    rnn_lstm_storm_tvalues.append([rnn_lstm_storm_t1_t, rnn_lstm_storm_t9_t, rnn_lstm_storm_t18_t])
    rnn_lstm_storm_pvalues.append([rnn_lstm_storm_t1_p, rnn_lstm_storm_t9_p, rnn_lstm_storm_t18_p])

    fullRNNStorms_vs_stormRNN_t1_t, fullRNNStorms_vs_stormRNN_t1_p = stats.ttest_ind(rmse_df["full_rnnstorms_t+1"],
                                                                                     rmse_df["storm_rnn_t+1"])
    fullRNNStorms_vs_stormRNN_t9_t, fullRNNStorms_vs_stormRNN_t9_p = stats.ttest_ind(rmse_df["full_rnnstorms_t+9"],
                                                                                     rmse_df["storm_rnn_t+9"])
    fullRNNStorms_vs_stormRNN_t18_t, fullRNNStorms_vs_stormRNN_t18_p = stats.ttest_ind(rmse_df["full_rnnstorms_t+18"],
                                                                                       rmse_df["storm_rnn_t+18"])
    fullRNNStorms_vs_stormRNN_tvalues.append([fullRNNStorms_vs_stormRNN_t1_t, fullRNNStorms_vs_stormRNN_t9_t,
                                              fullRNNStorms_vs_stormRNN_t18_t])
    fullRNNStorms_vs_stormRNN_pvalues.append([fullRNNStorms_vs_stormRNN_t1_p, fullRNNStorms_vs_stormRNN_t9_p,
                                              fullRNNStorms_vs_stormRNN_t18_p])

    fullLSTMStorms_vs_stormLSTM_t1_t, fullLSTMStorms_vs_stormLSTM_t1_p = stats.ttest_ind(rmse_df["full_lstmstorms_t+1"],
                                                                                         rmse_df["storm_lstm_t+1"])
    fullLSTMStorms_vs_stormLSTM_t9_t, fullLSTMStorms_vs_stormLSTM_t9_p = stats.ttest_ind(rmse_df["full_lstmstorms_t+9"],
                                                                                         rmse_df["storm_lstm_t+9"])
    fullLSTMStorms_vs_stormLSTM_t18_t, fullLSTMStorms_vs_stormLSTM_t18_p = stats.ttest_ind(rmse_df["full_lstmstorms_t+18"],
                                                                                           rmse_df["storm_lstm_t+18"])
    fullLSTMStorms_vs_stormLSTM_tvalues.append([fullLSTMStorms_vs_stormLSTM_t1_t, fullLSTMStorms_vs_stormLSTM_t9_t,
                                                fullLSTMStorms_vs_stormLSTM_t18_t])
    fullLSTMStorms_vs_stormLSTM_pvalues.append([fullLSTMStorms_vs_stormLSTM_t1_p, fullLSTMStorms_vs_stormLSTM_t9_p,
                                                fullLSTMStorms_vs_stormLSTM_t18_p])

    # save t-test results to dataframe
    ttest_cols = ["rnn_full_storm_t", "rnn_full_storm_p", "lstm_full_storm_t", "lstm_full_storm_p",
                  "rnn_lstm_full_t", "rnn_lstm_full_p", "rnn_lstm_storm_t", "rnn_lstm_storm_p",
                  "fullRNNStorms_vs_stormRNN_t", "fullRNNStorms_vs_stormRNN_p", "fullLSTMStorms_vs_stormLSTM_t",
                  "fullLSTMStorms_vs_stormLSTM_p"]

    ttest_df = pd.DataFrame([rnn_full_storm_tvalues[0], rnn_full_storm_pvalues[0],
                             lstm_full_storm_tvalues[0], lstm_full_storm_pvalues[0],
                             rnn_lstm_full_tvalues[0], rnn_lstm_full_pvalues[0],
                             rnn_lstm_storm_tvalues[0], rnn_lstm_storm_pvalues[0],
                             fullRNNStorms_vs_stormRNN_tvalues[0], fullRNNStorms_vs_stormRNN_pvalues[0],
                             fullLSTMStorms_vs_stormLSTM_tvalues[0], fullLSTMStorms_vs_stormLSTM_pvalues[0]])\
        .transpose()
    ttest_df.columns = ttest_cols
    ttest_df["forecast"] = ["t+1", "t+9", "t+18"]
    ttest_df = ttest_df.set_index("forecast")

    ttest_df.to_csv(os.path.join(out_folder, "ttest.csv"))

    # calculate means
    mean_list = []
    for i in col_list:
        col_mean = rmse_df[i].mean()
        mean_list.append(col_mean)

    mean_df = pd.DataFrame(mean_list).transpose()
    mean_df.columns = col_list

    mean_df.to_csv(os.path.join(out_folder, "means.csv"), index=False)

    # calculate confidence intervals
    upper_ci_list = []
    lower_ci_list = []
    for i in col_list:
        col_ci = stats.t.interval(0.95, len(rmse_df[i]) - 1, loc=np.mean(rmse_df[i]), scale=stats.sem(rmse_df[i]))
        upper_ci_list.append(col_ci[1])
        lower_ci_list.append(col_ci[0])

    # calculate error
    # rnn_rmse_t1_err = rnn_rmse_t1_mean - rnn_rmse_t1_ci[0]

    # save CIs to df
    ci_df = pd.DataFrame([lower_ci_list, upper_ci_list], columns=col_list, index=["lower", "upper"])
    ci_df.to_csv(os.path.join(out_folder, "CIs.csv"))
