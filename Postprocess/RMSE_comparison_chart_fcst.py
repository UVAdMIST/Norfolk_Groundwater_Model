"""
Written by Ben Bowes, 10-3-18

This script reads performance data on forecast test data of all wells and plots the RMSE and NSE in a bar chart.
Example from: https://stackoverflow.com/questions/22780563/group-labels-in-matplotlib-barchart-using-pandas-multiindex
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
import os


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='gray')
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index, level):
            lxpos = (pos + .5 * rpos)*scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale, ypos)
        ypos -= .1


def main():
    # set standard font size
    matplotlib.rcParams.update({'font.size': 8})

    # read data
    base_path = "C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results_fcst"
    comparison5_df = pd.read_csv(os.path.join(base_path, "comparison5_rmse.csv"))
    comparison5_df.set_index(['Well', 'Forecast'], inplace=True)

    comparison6_df = pd.read_csv(os.path.join(base_path, "comparison6_rmse.csv"))
    comparison6_df.set_index(['Well', 'Forecast'], inplace=True)

    comparison7_df = pd.read_csv(os.path.join(base_path, "comparison7_rmse.csv"))
    comparison7_df.set_index(['Well', 'Forecast'], inplace=True)

    comparison8_df = pd.read_csv(os.path.join(base_path, "comparison8_rmse.csv"))
    comparison8_df.set_index(['Well', 'Forecast'], inplace=True)

    # set up multi-index dataframes
    # group = ('MMPS-043', 'MMPS-125', 'MMPS-129', 'MMPS-153', 'MMPS-155', 'MMPS-170', 'MMPS-175')
    # subgroup = ('t+1', 't+9', 't+18')
    # obs = ('RNN', 'LSTM')
    index = pd.MultiIndex.from_tuples([('MMPS-043', 't+1'), ('MMPS-043', 't+9'), ('MMPS-043', 't+18'),
                                       ('MMPS-125', 't+1'), ('MMPS-125', 't+9'), ('MMPS-125', 't+18'),
                                       ('MMPS-129', 't+1'), ('MMPS-129', 't+9'), ('MMPS-129', 't+18'),
                                       ('MMPS-153', 't+1'), ('MMPS-153', 't+9'), ('MMPS-153', 't+18'),
                                       ('MMPS-155', 't+1'), ('MMPS-155', 't+9'), ('MMPS-155', 't+18'),
                                       ('MMPS-170', 't+1'), ('MMPS-170', 't+9'), ('MMPS-170', 't+18'),
                                       ('MMPS-175', 't+1'), ('MMPS-175', 't+9'), ('MMPS-175', 't+18')],
                                      names=['group', 'subgroup'])
    rmse_5 = np.asarray(comparison5_df[["RNN all data", "LSTM all data"]])
    rmse_6 = np.asarray(comparison6_df[["RNN storm data", "LSTM storm data"]])
    rmse_7 = np.asarray(comparison7_df[["RNN all data", "RNN storm data"]])
    rmse_8 = np.asarray(comparison8_df[["LSTM all data", "LSTM storm data"]])

    rmse5_df = pd.DataFrame(index=index)
    rmse5_df['RNN all data'] = rmse_5[:, 0]
    rmse5_df['LSTM all data'] = rmse_5[:, 1]

    rmse6_df = pd.DataFrame(index=index)
    rmse6_df['RNN storm data'] = rmse_6[:, 0]
    rmse6_df['LSTM storm data'] = rmse_6[:, 1]

    rmse7_df = pd.DataFrame(index=index)
    rmse7_df['RNN all data'] = rmse_7[:, 0]
    rmse7_df['RNN storm data'] = rmse_7[:, 1]

    rmse8_df = pd.DataFrame(index=index)
    rmse8_df['LSTM all data'] = rmse_8[:, 0]
    rmse8_df['LSTM storm data'] = rmse_8[:, 1]

    # create figure
    rmse_list = [rmse5_df, rmse6_df, rmse7_df, rmse8_df]
    fig_letter = ["A", "B", "C", "D"]

    fig = plt.figure(1, figsize=(6.5, 6.5))
    for i in range(0, len(rmse_list), 1):
        ax = plt.subplot(4, 1, i+1)
        ax.axvspan(2.5, 5.5, alpha=0.5, color='#A9A9A9')
        ax.axvspan(8.5, 11.5, alpha=0.5, color='#A9A9A9')
        ax.axvspan(14.5, 17.5, alpha=0.5, color='#A9A9A9')
        rmse_list[i].plot.bar(ax=ax, color=['k', 'white'], ec='k', stacked=False, )
        ax.set_xticklabels('')
        ax.set_xlabel('')
        ax.set_ylim(0, 1.5)
        ax.legend(loc=2)
        ax.text(19.75, 1.35, fig_letter[i])
        # ax.set_ylabel("RMSE (m)")

    label_group_bar_table(ax, rmse_list[i])
    fig.text(0.015, 0.5, "Mean RMSE (m)", horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, left=0.08)
    # plt.show()
    plt.savefig("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results_fcst/rmse_comparisons.png",
                dpi=300)


if __name__ == '__main__':
    main()
