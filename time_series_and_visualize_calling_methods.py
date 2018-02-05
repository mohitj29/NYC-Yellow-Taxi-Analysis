"""
Code containing Time series analysis for NYC taxi data
Author: Group 5, Fall 2017
"""

''' Importing all the lib'''
import time_series_and_visualize_module as tsv



'''Time Series class'''
ts = tsv.NYC_TS('TimeSeriesDataset.csv') # creating Time series class
# it will show you trend from 2009-2016 of pick up at a particular loaction(near time square)
ts.TimeSeries()
# it will show you the autocorrelation
ts.Autocorrelation()
# it will show you the Decomposition
ts.Decomposition()


''' Visulazation class'''
viz = tsv.Visualization()   #creating visualization class

#calling boxplot function and sending data loaction which will show you the distribution of duration of taxi ride
#viz.Duration_boxplot("final.csv")  # here i used a different file which is just sorted part of main file to make things run fast

#calling payment_histogram function and sending data loaction which will show you the method of payment used by customer
viz.payment_his("cash.csv") # here i used a different file which is just sorted part of main file to make things run fast



