# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 09:06:08 2017

@author: Ben Bowes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

#read raw data
df = pd.read_csv('C:/Users/Ben Bowes/Documents/HRSD GIS/Matlab_NARX/Prediction_Results/Comparisons/MMPS_170_96hr/20160903_hr10.csv',index_col = 'Datetime', parse_dates = True, infer_datetime_format = True)
df.reset_index(inplace=True)

#create blank figure
#fig = plt.figure(figsize = (3,3),dpi = 300)
fig, ax = plt.subplots()

ax.plot(df['Land_Surface'], 'k-.', label = 'Land Surface')
ax.plot(df['GWL_actual'], 'k', label = 'Actual')
ax.plot(df['GWL_observed'], 'k--', label = 'Forecast, Historical Data')
ax.plot(df['GWL_forecast'], 'k:', label = 'Forecast, Forecast Data')

ticks = np.arange(0, 18, 1)
ax.set_xticks(ticks)
ax.set_xticklabels(df['Datetime'][ticks].dt.strftime('%H:%M'), rotation='vertical')
ax.set_ylabel('GWL (ft)')
ax.set_xlabel('September 20, 2016')
legend = ax.legend(loc = 'best')


#display plot
plt.tight_layout()
plt.show()

#save plot for publication
fig.savefig('C:/Users/Ben Bowes/Documents/HRSD GIS/Matlab_NARX/Prediction_Results/Comparisons/MMPS_170_96hr/20160903_hr10.tif',dpi = 300)