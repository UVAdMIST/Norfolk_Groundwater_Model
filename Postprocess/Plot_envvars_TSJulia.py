import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

start_date="2016-09-17 00:00:00"
stop_date="2016-09-25 00:00:00"

#read raw data
MMPS_043 = pd.read_csv("./Observed_data/MMPS_043.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_043 = MMPS_043.loc[start_date:stop_date]
MMPS_043.reset_index(inplace=True)

MMPS_125 = pd.read_csv("./Observed_data/MMPS_125.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_125 = MMPS_125.loc[start_date:stop_date]
MMPS_125.reset_index(inplace=True)

MMPS_129 = pd.read_csv("./Observed_data/MMPS_129.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_129 = MMPS_129.loc[start_date:stop_date]
MMPS_129.reset_index(inplace=True)

MMPS_153 = pd.read_csv("./Observed_data/MMPS_153.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_153 = MMPS_153.loc[start_date:stop_date]
MMPS_153.reset_index(inplace=True)

MMPS_155 = pd.read_csv("./Observed_data/MMPS_155.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_155 = MMPS_155.loc[start_date:stop_date]
MMPS_155.reset_index(inplace=True)

MMPS_170 = pd.read_csv("./Observed_data/MMPS_170.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_170 = MMPS_170.loc[start_date:stop_date]
MMPS_170.reset_index(inplace=True)

MMPS_175 = pd.read_csv("./Observed_data/MMPS_175.csv", index_col='Datetime',
                       parse_dates=True, infer_datetime_format=True)
MMPS_175 = MMPS_175.loc[start_date:stop_date]
MMPS_175.reset_index(inplace=True)

#create blank figure
fig, axarr = plt.subplots(2, 2, sharey=False,sharex=False, figsize=(8, 6))


#ticks = np.arange(0, 18, 1)

ax1 = axarr[0,1]
ax1.plot(MMPS_043['GWL'], 'k-.', label = 'Land Surface')
ax1.plot(MMPS_043['Tide'], 'k', label = 'Observed')
# ax1.set_ylim(ymin=-1.5, ymax=10.0)
start, end = ax1.get_xlim()
x_ticks = np.arange(0, end, 24)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(MMPS_043['Datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')

ax12=ax1.twinx()

# ax12.set_ylim(ax12.get_ylim()[:: 0.5])
# plt.ylim(4,0)

x2_ticks_2 = np.arange(0, len(MMPS_043['Precip.Avg']), 1)
ax12.bar(x2_ticks_2, MMPS_043['Precip.Avg'])
ax12.set_ylim(ymin=0.0, ymax=4.0)
plt.gca().invert_yaxis()
ax12.set_ylabel("Total Hourly Precip. (in)")

# fig.canvas.draw()

# ax1.set_ylabel('GWL (ft)')
#ax1.text(0.5,5.5, 'A')

# # ax2 = axarr[0,1]
# # ax2.plot(MMPS_125['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax2.plot(MMPS_125['GWL_actual'], 'k', label = 'Observed')
# # ax2.plot(MMPS_125['GWL_observed'], 'k--', label = 'Hindcast')
# # ax2.plot(MMPS_125['GWL_forecast'], 'k:', label = 'Forecast')
# # ax2.set_ylim(0,5)
# # ax2.set_xticks(ticks)
# # # ax2.set_xticklabels(MMPS_125['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # ax2.set_xticklabels([])
# # # ax2.set_ylabel('GWL (ft)')
# # ax2.text(0.5, 3.75, 'B')
# #
# # ax3 = axarr[1,0]
# # ax3.plot(MMPS_129['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax3.plot(MMPS_129['GWL_actual'], 'k', label = 'Observed')
# # ax3.plot(MMPS_129['GWL_observed'], 'k--', label = 'Hindcast')
# # ax3.plot(MMPS_129['GWL_forecast'], 'k:', label = 'Forecast')
# # ax3.set_xticks(ticks)
# # # ax3.set_xticklabels(MMPS_129['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # ax3.set_xticklabels([])
# # # ax3.set_ylabel('GWL (ft)')
# # ax3.text(0.5, 12.5, 'C')
# #
# # ax4 = axarr[1,1]
# # ax4.plot(MMPS_153['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax4.plot(MMPS_153['GWL_actual'], 'k', label = 'Observed')
# # ax4.plot(MMPS_153['GWL_observed'], 'k--', label = 'Hindcast')
# # ax4.plot(MMPS_153['GWL_forecast'], 'k:', label = 'Forecast')
# # ax4.set_xticks(ticks)
# # # ax4.set_xticklabels(MMPS_153['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # ax4.set_xticklabels([])
# # # ax4.set_ylabel('GWL (ft)')
# # ax4.text(0.5, 8.75, 'D')
# #
# # ax5 = axarr[2,0]
# # ax5.plot(MMPS_155['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax5.plot(MMPS_155['GWL_actual'], 'k', label = 'Observed')
# # ax5.plot(MMPS_155['GWL_observed'], 'k--', label = 'Hindcast')
# # ax5.plot(MMPS_155['GWL_forecast'], 'k:', label = 'Forecast')
# # ax5.set_ylim(1,6)
# # ax5.set_xticks(ticks)
# # # ax5.set_xticklabels(MMPS_155['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # ax5.set_xticklabels([])
# # # ax5.set_ylabel('GWL (ft)')
# # ax5.text(0.5, 4.75, 'E')
# #
# # ax6 = axarr[2,1]
# # ax6.plot(MMPS_170['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax6.plot(MMPS_170['GWL_actual'], 'k', label = 'Observed')
# # ax6.plot(MMPS_170['GWL_observed'], 'k--', label = 'Hindcast')
# # ax6.plot(MMPS_170['GWL_forecast'], 'k:', label = 'Forecast')
# # ax6.set_xticks(ticks)
# # ax6.set_xticklabels(MMPS_170['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # # ax6.set_ylabel('GWL (ft)')
# # ax6.text(0.5, 6.5, 'F')
# #
# # ax7 = axarr[3,0]
# # ax7.plot(MMPS_175['Land_Surface'], 'k-.', label = 'Land Surface')
# # ax7.plot(MMPS_175['GWL_actual'], 'k', label = 'Observed')
# # ax7.plot(MMPS_175['GWL_observed'], 'k--', label = 'Hindcast')
# # ax7.plot(MMPS_175['GWL_forecast'], 'k:', label = 'Forecast')
# # ax7.set_ylim(1,9)
# # ax7.set_xticks(ticks)
# # ax7.set_xticklabels(MMPS_175['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
# # # ax7.set_ylabel('GWL (ft)')
# # ax7.text(0.5, 6.75, 'G')
# # legend = ax7.legend(bbox_to_anchor=(1.78, 0.75))
# #
# # ax8 = axarr[3,1]
# # ax8.axis('off')
#
# #display plot
plt.tight_layout()
# plt.subplots_adjust(hspace=0.20)
# # fig.text(0.01, 0.53, 'GWL (ft)', va='center', rotation='vertical')
plt.show()
#
# #save plot for publication
# # fig.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Matlab_NARX/Prediction_Results/Comparisons/TSJulia.png',dpi = 300)