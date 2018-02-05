"""
Code containing class and methods for taxi locations density at different times.
Author : Group 5 , fall 2017
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import lognorm
import random

import warnings
def fxn():
    warnings.warn("deprecated",DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import pandas as pd
import statsmodels.formula.api as smf 
import statsmodels.api as sm


#preprocessing
#Reading the data
dftaxi = pd.read_csv('./2015.csv')

#convert to date
dftaxi['date'] = pd.to_datetime(dftaxi.date)

dftaxi['year'] = pd.DatetimeIndex(dftaxi["date"]).year
dftaxi['month'] = pd.DatetimeIndex(dftaxi["date"]).month
dftaxi['day'] = pd.DatetimeIndex(dftaxi["date"]).day
dftaxi['dayofyear'] = pd.DatetimeIndex(dftaxi["date"]).dayofyear
dftaxi['weekofyear'] = pd.DatetimeIndex(dftaxi["date"]).weekofyear
dftaxi['dayofweek'] = pd.DatetimeIndex(dftaxi["date"]).dayofweek
dftaxi['weekday'] = pd.DatetimeIndex(dftaxi["date"]).weekday
dftaxi['quarter'] = pd.DatetimeIndex(dftaxi["date"]).quarter
dftaxi['hour'] = pd.DatetimeIndex(dftaxi["time"]).hour

print(dftaxi.shape)
print(dftaxi.head())

#reomving locations which point to value zero
dftaxi = dftaxi[dftaxi.lat >= 0]
dftaxi = dftaxi[dftaxi.long <= 0]

# Remove coordinate outliers
dftaxi = dftaxi[dftaxi['long'] <= -73.75]
dftaxi = dftaxi[dftaxi['long'] >= -74.03]
dftaxi = dftaxi[dftaxi['lat'] <= 40.85]
dftaxi = dftaxi[dftaxi['lat'] >= 40.63]

print(dftaxi.shape)
print(dftaxi.head())

#histogram
plt.hist(dftaxi.num_pickups,normed=True, bins=5)
plt.ylabel('Frequency')
plt.show()

#histogram
plt.bar(dftaxi.num_pickups,dftaxi.year)
plt.ylabel('Year')
plt.show()

## clean data : no Nan and zero values
print(dftaxi.describe().transpose())

#heatmap
cleanup_nums = {"dayofweek": {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                                  4: "Friday", 5: "Saturday", 6:"Sunday" }}

dftaxi.replace(cleanup_nums, inplace=True)

#dftaxi.reset_index().pivot_table(values=3, index=[0, 1], columns=2, aggfunc='mean')

new_df = pd.pivot_table(dftaxi,index=['hour'], columns = 'dayofweek', values = "num_pickups",aggfunc='sum')



# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
ax.set_title('Heat map of Hour and Weekday Vs Number of pickups', fontsize = 20)
sns.heatmap(new_df, annot=True, linewidths=.5, ax=ax)
plt.show()




# define the figure with 2 subplots
fig,ax = plt.subplots(1,2,figsize = (15,4))

#if data is skewed negative binomial will perform better than poisson

# histogram of the number of pickups
dftaxi.num_pickups.hist(bins=30,ax=ax[0])
ax[0].set_xlabel('Num of Pickups')
ax[0].set_ylabel('Count')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of Pickups - Normal Scale')

# create a vector to hold num of pickups
v = dftaxi.num_pickups 

# plot the histogram with 30 bins
v[~((v-v.median()).abs()>3*v.std())].hist(bins=30,ax=ax[1]) 
ax[1].set_xlabel('Num of pickups')
ax[1].set_ylabel('Count')
ax[1].set_title('Histogram of Num of pickups - Scaled')

# apply a lognormal fit. Use the mean of trip distance as the scale parameter
scatter,loc,mean = lognorm.fit(dftaxi.num_pickups.values,
                               scale=dftaxi.num_pickups.mean(),
                               loc=0)
pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)
ax[1].plot(np.arange(0,12,.1),600000*pdf_fitted,'r') 
ax[1].legend(['data','lognormal fit'])


plt.show()

#Pickup density
pickup_xaxis = dftaxi.long
pickup_yaxis = dftaxi.lat
sns.set_style('white')
fig, ax = plt.subplots(figsize=(11, 12))
ax.set_axis_bgcolor('black')
ax.scatter(pickup_xaxis, pickup_yaxis, s=7, color='lightpink', alpha=0.7)
ax.set_xlim([-74.03, -73.90])
ax.set_ylim([40.63, 40.85])
ax.set_xlabel("Longitude", fontsize = 12)
ax.set_ylabel("Latitude", fontsize = 12)
ax.set_title('Number of Pickups as pink dots on NYC', fontsize = 20)
plt.show()


#poisson
model = smf.glm(formula = "num_pickups ~ year + month + lat +long +dayofweek +day+quarter+hour", data=dftaxi, family=sm.families.Poisson()).fit()

print("Poisson Model Summary")


print(model.summary())



#print(model.fittedvalues)

print("\n")

#include accuracy measures

print("RMSE for Poisson Regression Model : ",sm.tools.eval_measures.rmse(dftaxi.num_pickups, model.fittedvalues, axis=0))


#negative binomial

model = smf.glm(formula = "num_pickups ~ year + month + lat +long +dayofweek +day+quarter+hour", data=dftaxi, family=sm.families.NegativeBinomial()).fit()

print("Negative Binomial  Model Summary")


print(model.summary())

#print(model.fittedvalues)

#include accuracy measures

print("RMSE for Negative Binomial Regression Model : ",sm.tools.eval_measures.rmse(dftaxi.num_pickups, model.fittedvalues, axis=0))
