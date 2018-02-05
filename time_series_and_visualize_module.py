import pandas as pd
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

class NYC_TS(): #creating a class for time series
    def __init__(self,path):
        self.series = Series.from_csv(path, header=0) #importing the document
    def TimeSeries(self): # function to plot the pickups near time square from 2009-2016
        self.series.plot()
        pyplot.show() # it will show original time series plot of the data
    def Autocorrelation(self):      # To plot the auto correlation
        autocorrelation_plot(self.series)
        pyplot.show()
    def Decomposition(self): # to plot the Decomposition
        decomposition = seasonal_decompose(self.series) 
        trend = decomposition.trend  #It will decompose the original plot into trends
        seasonal = decomposition.seasonal # It will decompose the original plot into seasonality 
        residual = decomposition.resid #It will decompose the original plot into irregular components 
        pyplot.subplot(411)
        pyplot.plot(self.series, label='Original')
        pyplot.legend(loc='best')
        pyplot.subplot(412)
        pyplot.plot(trend, label='Trend')
        pyplot.legend(loc='best')                             # It will show all the plot
        pyplot.subplot(413)
        pyplot.plot(seasonal,label='Seasonality')
        pyplot.legend(loc='best')
        pyplot.subplot(414)
        pyplot.plot(residual, label='Residuals')
        pyplot.legend(loc='best')
        pyplot.tight_layout()
        pyplot.show()

class Visualization(): # class for visulization of data
    def Duration_boxplot(self,path): # function to create boxplot for duration of ride
        file = pd.read_csv(path) #import file
        result = []  #creating a list for duration in minutes
        count = 0
        try:
            for i in range(len(file.duration)): #converting duration time into integer
               # print(i)
                if file.duration[i] != '###############################################################################################################################################################################################################################################################':
                    h,m,s = file.duration[i].split(':')
                    result.append(int((int(h) * 3600 + int(m) * 60 + int(s))/60))
        except Exception:
            count +=1
        
        pyplot.boxplot(result,showfliers=False) #plot actual box plot
        pyplot.xlabel("Duration of Ride") 
        pyplot.ylabel("Minutes")
        pyplot.show() # show plot
        pyplot.boxplot(result) #detailed boxplot
        pyplot.xlabel("Duration of Ride") 
        pyplot.ylabel("Minutes")
        pyplot.show()
        
    def payment_his(self,path): #function for analyse payment type
        file = pd.read_csv(path, header=0) #import file
        pyplot.hist(file.payment_type) #ploting histogram
        pyplot.xlabel("Method of payment")
        pyplot.show() # show the graph
