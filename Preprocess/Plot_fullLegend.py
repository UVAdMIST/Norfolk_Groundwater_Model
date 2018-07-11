import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

storm_data = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Norfolk_averaged_data.csv",
                         index_col="Datetime", parse_dates=True, infer_datetime_format=True)

start_date = "2016-09-17 00:00:00"
stop_date = "2016-09-28 00:00:00"
storm = storm_data.loc[start_date:stop_date]
storm.reset_index(inplace=True)

ax = storm[["GWL", "Tide"]].plot(color=["k", "k"], style=["--", ":"], legend=None)
start, end = ax.get_xlim()
ticks = np.arange(0, end, 24)  # (start,stop,increment)
ax2 = ax.twinx()
# ax2.set_ylim(ymax=1, ymin=0)
# ax.set_ylim(ymax=4.5, ymin=-1.5)
ax2.invert_yaxis()
storm["Precip."].plot.bar(ax=ax2, color="k")
ax2.set_xticks([])
ax.set_xticks(ticks)
ax.set_xticklabels(storm.loc[ticks, 'Datetime'].dt.strftime('%Y-%m-%d'), rotation='vertical')
ax.set_ylabel("Hourly Avg GW/Tide Level (ft)")
# ax.set_ylim(ymax=13,ymin=-3)
ax2.set_ylabel("Total Hourly Precip. (in)")
# plt.title(site)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2)  # location: 0=best, 9=top center

plt.tight_layout()
# plt.show()

# save plot for publication
plt.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Presentation Images/Plots/Floods_GWL_comparisons/20160919_bw_averaged.png',
            dpi=300)
