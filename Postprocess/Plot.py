import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

storm_data=pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_129.csv",index_col="Datetime",parse_dates=True, infer_datetime_format=True)
site="MMPS-118"
start_date="2016-09-01 00:00:00"
stop_date="2016-09-30 00:00:00"
storm=storm_data.ix[start_date:stop_date]

storm.reset_index(inplace=True)
# print(storm)
ax = storm[["GWL","Tide"]].plot(color=["k","k"], style=["--",":"])
start, end = ax.get_xlim()
ticks = np.arange(0, end, 24) #(start,stop,increment)
ax2=ax.twinx()
ax2.set_ylim(ymax=1, ymin=0)
# ax.set_ylim(ymax=4.5, ymin=-1.5)
ax2.invert_yaxis()
storm["Precip."].plot.bar(ax = ax2, color="k")
ax2.set_xticks([])
ax.set_xticks(ticks)
ax.set_xticklabels(storm.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax.set_ylabel("Hourly Avg GW/Tide Level (ft)")
# ax.set_ylim(ymax=13,ymin=-3)
ax2.set_ylabel("Total Hourly Precip. (in)")
# plt.title(site)
plt.tight_layout()
plt.show()

#save plot for publication
# plt.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Plots/Floods_GWL_comparisons/20160204_avgdata.png',dpi = 300)