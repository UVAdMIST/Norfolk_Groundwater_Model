import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

start_date="2016-09-17 00:00:00"
stop_date="2016-09-28 00:00:00"

# read raw data
MMPS_043 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_043_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_043 = MMPS_043.loc[start_date:stop_date]
MMPS_043.reset_index(inplace=True)

MMPS_125 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_125_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_125 = MMPS_125.loc[start_date:stop_date]
MMPS_125.reset_index(inplace=True)

MMPS_129 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_129_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_129 = MMPS_129.loc[start_date:stop_date]
MMPS_129.reset_index(inplace=True)

MMPS_153 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_153_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_153 = MMPS_153.loc[start_date:stop_date]
MMPS_153.reset_index(inplace=True)

MMPS_155 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_155_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_155 = MMPS_155.loc[start_date:stop_date]
MMPS_155.reset_index(inplace=True)

MMPS_170 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_170_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_170 = MMPS_170.loc[start_date:stop_date]
MMPS_170.reset_index(inplace=True)

MMPS_175 = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_175_no_blanks.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_175 = MMPS_175.loc[start_date:stop_date]
MMPS_175.reset_index(inplace=True)

#create blank figure
fig, axarr = plt.subplots(4, 2, sharey=False,sharex=False, figsize=(8, 6))


ax1 = axarr[0,0]
ax1.plot(MMPS_043['GWL'], 'k-.', label = 'GWL')
ax1.plot(MMPS_043['Tide'], 'k:', label = 'Tide')
ax1.set_ylim(ymin=-2, ymax=4.0)
start, end = ax1.get_xlim()
x_ticks = np.arange(0, end, 24)
ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax12=ax1.twinx()
x2_ticks_2 = np.arange(0, len(MMPS_043['Precip.Avg']), 1)
ax12.bar(x2_ticks_2, MMPS_043['Precip.Avg'], color='k', label='Precip.')
ax12.set_xticklabels([])
ax12.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax12.set_yticklabels([])
ax1.text(190, 3.25, 'MMPS-043 (7.3ft)')

ax2 = axarr[0,1]
ax2.plot(MMPS_125['GWL'], 'k-.', label = 'GWL')
ax2.plot(MMPS_125['Tide'], 'k:', label = 'Tide')
ax2.set_ylim(ymin=-2, ymax=6.0)
# ax2.set_yticklabels([])
start, end = ax2.get_xlim()
x_ticks = np.arange(0, end, 24)
ax2.set_xticks(x_ticks)
# ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax22=ax2.twinx()
# ax22.invert_yaxis()
x22_ticks_2 = np.arange(0, len(MMPS_125['Rain']), 1)
ax22.bar(x22_ticks_2, MMPS_125['Rain'], color='k', label='Precip.')
ax22.set_xticklabels([])
ax22.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax2.text(190,4.9, 'MMPS-125 (4.1ft)')

ax3 = axarr[1,0]
ax3.plot(MMPS_129['GWL'], 'k-.', label = 'GWL')
ax3.plot(MMPS_129['Tide'], 'k:', label = 'Tide')
ax3.set_ylim(ymin=-2, ymax=14.0)
start, end = ax3.get_xlim()
x_ticks = np.arange(0, end, 24)
ax3.set_xticks(x_ticks)
# ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax32=ax3.twinx()
x32_ticks_2 = np.arange(0, len(MMPS_129['Precip.']), 1)
ax32.bar(x32_ticks_2, MMPS_129['Precip.'], color='k', label='Precip.')
ax32.set_xticklabels([])
ax32.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax32.set_yticklabels([])
ax3.text(185,11.75, 'MMPS-129 (14.3ft)')

ax4 = axarr[1,1]
ax4.plot(MMPS_153['GWL'], 'k-.', label = 'GWL')
ax4.plot(MMPS_153['Tide'], 'k:', label = 'Tide')
ax4.set_ylim(ymin=-2, ymax=7.0)
start, end = ax4.get_xlim()
x_ticks = np.arange(0, end, 24)
ax4.set_xticks(x_ticks)
# ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax42=ax4.twinx()
x42_ticks_2 = np.arange(0, len(MMPS_153['Precip.Avg']), 1)
ax42.bar(x42_ticks_2, MMPS_153['Precip.Avg'], color='k', label='Precip.')
ax42.set_xticklabels([])
ax42.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax4.text(185, 5.5, 'MMPS-153 (10.6ft)')

ax5 = axarr[2,0]
ax5.plot(MMPS_155['GWL'], 'k-.', label = 'GWL')
ax5.plot(MMPS_155['Tide'], 'k:', label = 'Tide')
ax5.set_ylim(ymin=-2, ymax=6.0)
start, end = ax5.get_xlim()
x_ticks = np.arange(0, end, 24)
ax5.set_xticks(x_ticks)
# ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax52=ax5.twinx()
x52_ticks_2 = np.arange(0, len(MMPS_155['Precip.Avg']), 1)
ax52.bar(x52_ticks_2, MMPS_155['Precip.Avg'], color='k', label='Precip.')
ax52.set_xticklabels([])
ax52.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax52.set_yticklabels([])
ax5.text(190,4.9, 'MMPS-155 (5.6ft)')

ax6 = axarr[2,1]
ax6.plot(MMPS_170['GWL'], 'k-.', label = 'GWL')
ax6.plot(MMPS_170['Tide'], 'k:', label = 'Tide')
ax6.set_ylim(ymin=-2, ymax=5.0)
start, end = ax6.get_xlim()
x_ticks = np.arange(0, end, 24)
ax6.set_xticks(x_ticks)
ax6.set_xticklabels(MMPS_170['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax62=ax6.twinx()
x62_ticks_2 = np.arange(0, len(MMPS_170['Precip.']), 1)
ax62.bar(x62_ticks_2, MMPS_170['Precip.'], color='k', label='Precip.')
# ax62.set_xticklabels([])
ax62.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax6.text(190,4.1, 'MMPS-170 (7.7ft)')

ax7 = axarr[3,0]
ax7.plot(MMPS_175['GWL'], 'k-.', label = 'GWL')
ax7.plot(MMPS_175['Tide'], 'k:', label = 'Tide')
ax7.set_ylim(ymin=-2, ymax=6.0)
start, end = ax7.get_xlim()
x_ticks = np.arange(0, end, 24)
ax7.set_xticks(x_ticks)
ax7.set_xticklabels(MMPS_175['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax72=ax7.twinx()
x72_ticks_2 = np.arange(0, len(MMPS_175['Precip.']), 1)
ax72.bar(x72_ticks_2, MMPS_175['Precip.'], color='k', label='Precip.')
# ax72.set_xticklabels([])
ax72.set_ylim(ymin=0.0, ymax=2.5)
plt.gca().invert_yaxis()
ax7.text(190,5, 'MMPS-175 (8.4ft)')
lines, labels = ax7.get_legend_handles_labels()
lines2, labels2 = ax72.get_legend_handles_labels()
legend = ax7.legend(lines+lines2, labels+labels2, bbox_to_anchor=(1.8, 0.2))

ax8 = axarr[3,1]
ax8.axis('off')
#
# #display plot
plt.tight_layout()
plt.subplots_adjust(right=0.94, wspace=0.15, hspace=0.20)
fig.text(0.01, 0.525, 'Hourly GW/Tide Level (ft)', va='center', rotation='vertical')
fig.text(0.975, 0.525, 'Total Hourly Precip. (in)', va='center', rotation='vertical')
# fig.text(0.75, 0.12, 'Land Surface\nElevation (ft.):\nA = 7.3\nB = 4.1\nC = 14.3'
#                      '\nD = 10.6\nE = 5.6\nF = 7.7\nG = 8.4', va='center')
# plt.show()

#save plot for publication
fig.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Paper Figures/TSJulia_envvars.png',dpi = 300)