"""
Used examples from SO:
https://stackoverflow.com/questions/22780563/group-labels-in-matplotlib-barchart-using-pandas-multiindex
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np


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


# set standard font size
matplotlib.rcParams.update({'font.size': 8})

# read data
raw_rmse_df = pd.read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results/"
                          "full_testset_rmse.csv")
raw_rmse_df.set_index(['Well', 'Forecast'], inplace=True)

raw_nse_df = pd.read_csv("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results/"
                         "full_testset_nse.csv")
raw_nse_df.set_index(['Well', 'Forecast'], inplace=True)

# set up multi-index dataframes
group = ('MMPS-043', 'MMPS-125', 'MMPS-129', 'MMPS-153', 'MMPS-155', 'MMPS-170', 'MMPS-175')
subgroup = ('t+1', 't+9', 't+18')
obs = ('RNN', 'LSTM')
index = pd.MultiIndex.from_tuples([('MMPS-043', 't+1'), ('MMPS-043', 't+9'), ('MMPS-043', 't+18'),
                                   ('MMPS-125', 't+1'), ('MMPS-125', 't+9'), ('MMPS-125', 't+18'),
                                   ('MMPS-129', 't+1'), ('MMPS-129', 't+9'), ('MMPS-129', 't+18'),
                                   ('MMPS-153', 't+1'), ('MMPS-153', 't+9'), ('MMPS-153', 't+18'),
                                   ('MMPS-155', 't+1'), ('MMPS-155', 't+9'), ('MMPS-155', 't+18'),
                                   ('MMPS-170', 't+1'), ('MMPS-170', 't+9'), ('MMPS-170', 't+18'),
                                   ('MMPS-175', 't+1'), ('MMPS-175', 't+9'), ('MMPS-175', 't+18')],
                                  names=['group', 'subgroup'])
rmse_values = np.asarray(raw_rmse_df[["RNN", "LSTM"]])
nse_values = np.asarray(raw_nse_df[["RNN", "LSTM"]])

rmse_df = pd.DataFrame(index=index)
rmse_df['RNN'] = rmse_values[:, 0]
rmse_df['LSTM'] = rmse_values[:, 1]

nse_df = pd.DataFrame(index=index)
nse_df['RNN'] = nse_values[:, 0]
nse_df['LSTM'] = nse_values[:, 1]

# create figure
plt.figure(1, figsize=(6.5, 4.5))

ax1 = plt.subplot(211)
rmse_df.plot.bar(ax=ax1, color=['k', 'white'], ec='k', stacked=False)
ax1.set_xticklabels('')
ax1.set_xlabel('')
ax1.set_ylabel("RMSE (m)")

ax2 = plt.subplot(212)
nse_df.plot.bar(ax=ax2, color=['k', 'white'], ec='k', legend=None, stacked=False)
ax2.set_xticklabels('')
ax2.set_xlabel('')
ax2.set_ylim(0.5, 1)
ax2.set_ylabel("NSE")
label_group_bar_table(ax2, nse_df)

plt.tight_layout()
plt.subplots_adjust(bottom=0.095)
plt.show()
# plt.savefig("C:/Users/Ben Bowes/PycharmProjects/Tensorflow/rnn_lstm_comparison_results/full_testset_metrics.png", dpi=300)
