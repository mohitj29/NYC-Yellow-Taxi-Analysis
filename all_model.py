"""
Code containing class and methods for fare tip analysis and taxi locations at different times.
Author : Group 5 , fall 2017
"""

#importing required predefined libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import statsmodels.formula.api as smf 
import statsmodels.api as sm
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import geohash
import random
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import lognorm
from math import sqrt

#Code to avoid warning messages
import warnings
#warnings.filterwarnings("ignore")
def fxn():
    warnings.warn("deprecated",DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#Class for taxi density model prediction and visalization analysis    
class TaxiDensity:
    #constructor - loading the data frame
    def __init__(self,filepath):
        self.dftaxi=pd.read_csv(filepath)

    #Data Preprocessing
    def dataPreProcessing(self):
        #converting the data type to date format for analaysis
        self.dftaxi['date'] = pd.to_datetime(self.dftaxi.date)

        self.dftaxi['year'] = pd.DatetimeIndex(self.dftaxi["date"]).year
        self.dftaxi['month'] = pd.DatetimeIndex(self.dftaxi["date"]).month
        self.dftaxi['day'] = pd.DatetimeIndex(self.dftaxi["date"]).day
        self.dftaxi['dayofyear'] = pd.DatetimeIndex(self.dftaxi["date"]).dayofyear
        self.dftaxi['weekofyear'] = pd.DatetimeIndex(self.dftaxi["date"]).weekofyear
        self.dftaxi['dayofweek'] = pd.DatetimeIndex(self.dftaxi["date"]).dayofweek
        self.dftaxi['weekday'] = pd.DatetimeIndex(self.dftaxi["date"]).weekday
        self.dftaxi['quarter'] = pd.DatetimeIndex(self.dftaxi["date"]).quarter

        #removing locations which point to value zero
        self.dftaxi = self.dftaxi[self.dftaxi.lat >= 0]
        self.dftaxi = self.dftaxi[self.dftaxi.long <= 0]

        # Remove coordinate outliers which are beyond NYC region
        self.dftaxi = self.dftaxi[self.dftaxi['long'] <= -73.75]
        self.dftaxi = self.dftaxi[self.dftaxi['long'] >= -74.03]
        self.dftaxi = self.dftaxi[self.dftaxi['lat'] <= 40.85]
        self.dftaxi = self.dftaxi[self.dftaxi['lat'] >= 40.63]

    #method to display the cleaned dataset with summary
    def dataDisplay(self):
        #Displaying the cleaned data
        print("===============================================")
        print(self.dftaxi.shape)
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Initial few lines of the cleaned data :")
        print(self.dftaxi.head())
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print(self.dftaxi.info())
        print("PREPROCESSING OF DATA IS COMPLETE!")
        print("-----------------------------------------------")
        ## clean data : no Nan and zero values
        print(self.dftaxi.describe().transpose())

    def dataHistogram(self):
        #Code for plotting the histogram of number of pickups
        plt.hist(self.dftaxi.num_pickups,normed=True, bins=5)
        plt.ylabel('Frequency')
        plt.title("Unscaled - Number of Pickups")
        plt.show()
        # define the figure with 2 subplots
        fig,ax = plt.subplots(1,2,figsize = (15,4))
        print("the unscaled graph is not representative and hence we go for scaling ")
        #if data is skewed negative binomial will perform better than poisson
        # histogram of the number of pickups
        self.dftaxi.num_pickups.hist(bins=30,ax=ax[0])
        ax[0].set_xlabel('Num of Pickups')
        ax[0].set_ylabel('Count')
        ax[0].set_yscale('log')
        ax[0].set_title('Histogram of Pickups - Normal Scale')

        # create a vector to hold num of pickups
        v = self.dftaxi.num_pickups 

        # plot the histogram with 30 bins
        v[~((v-v.median()).abs()>3*v.std())].hist(bins=30,ax=ax[1]) 
        ax[1].set_xlabel('Num of pickups')
        ax[1].set_ylabel('Count')
        ax[1].set_title('Histogram of Num of pickups - Scaled')
        print("A scaled graph is being plotted instead...!")
        print("\n")
        # apply a lognormal fit. Use the mean of trip distance as the scale parameter
        scatter,loc,mean = lognorm.fit(self.dftaxi.num_pickups.values,scale=self.dftaxi.num_pickups.mean(),loc=0)
        pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)
        ax[1].plot(np.arange(0,12,.1),600000*pdf_fitted,'r') 
        ax[1].legend(['data','lognormal fit'])
        plt.show()

    #Pickup density - graphical display over NYC area
    def DensityGraph(self):
        pickup_xaxis = self.dftaxi.long
        pickup_yaxis = self.dftaxi.lat
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(11, 12))
        ax.set_axis_bgcolor('black')
        ax.scatter(pickup_xaxis, pickup_yaxis, s=7, color='mediumaquamarine', alpha=0.7)
        ax.set_xlim([-74.03, -73.90])
        ax.set_ylim([40.63, 40.85])
        ax.set_xlabel("Longitude", fontsize = 12)
        ax.set_ylabel("Latitude", fontsize = 12)
        ax.set_title('Number of Pickups as green dots on NYC', fontsize = 20)
        plt.show()       

    #Heat Maps Visualization - based on number of pickups
    def DensityHeatMapGraph(self):
        print(" A heat map is being plotted for pickup density analysis...")
        print("\n")
        print("the first heat map is plotted based on Quarter and Day of week for pickup density")
        print("\n")
        #heat map for num of pickups for day of month
        cleanup_nums = {"dayofweek": {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",4: "Friday", 5: "Saturday", 6:"Sunday" }}
        self.dftaxi.replace(cleanup_nums, inplace=True)
        #utilizing the table function to filter values as needed.    
        new_df = pd.pivot_table(self.dftaxi,index=['quarter'], columns = 'dayofweek', values = "num_pickups",aggfunc='sum')
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(new_df, annot=True, linewidths=.5, ax=ax)
        plt.show()
        print("The second heat map is for month and year ")
        #replacing values
        cleanup_nums = {"dayofweek": {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3,"Friday":4, "Saturday":5, "Sunday":6 }}
        self.dftaxi.replace(cleanup_nums, inplace=True)

        #heat map for num of pickups for day of month
        new_df = pd.pivot_table(self.dftaxi,index=['month'], columns = 'year', values = "num_pickups",aggfunc='sum')
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(new_df, annot=True, linewidths=.5, ax=ax)
        plt.show()

    def RegressionModel(self):
        #poisson regression
        model = smf.glm(formula = "num_pickups ~ year + month + lat +long +dayofweek +day+quarter", data=self.dftaxi, family=sm.families.Poisson()).fit()
        print("Poisson Model Summary")
        print(model.summary())
        print("\n")
        #RMSE
        print("RMSE for Poisson Regression Model : ",sm.tools.eval_measures.rmse(self.dftaxi.num_pickups, model.fittedvalues, axis=0))
        print("-------------------------------------------------")
        #negative binomial regression
        model = smf.glm(formula = "num_pickups ~ year + month + lat +long +dayofweek +day+quarter", data=self.dftaxi, family=sm.families.NegativeBinomial()).fit()
        print("Negative Binomial  Model Summary")
        print(model.summary())
        print("\n")
        #RMSE
        print("RMSE for Negative Binomial Regression Model : ",sm.tools.eval_measures.rmse(self.dftaxi.num_pickups, model.fittedvalues, axis=0))
        print("-------------------------------------------------")
        

     # Funtion for cross-validation over a grid of parameters
    def validation_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None, verbose=0):
        if score_func:
            gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
        else:
            gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, verbose=verbose)
        gs.fit(X, y)
        print ("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_, gs.scorer_)
        print ("Best score: ", gs.best_score_)
        best = gs.best_estimator_
        return best

    #Method to build a Random forest regression model.
    def RandomForest(self):
        #train test split
        itrain, itest = train_test_split(range(self.dftaxi.shape[0]), train_size=0.8)
        mask=np.ones(self.dftaxi.shape[0], dtype='int')
        mask[itrain]=1
        mask[itest]=0
        mask = (mask==1)
        print(mask[:10])

        # Split off the features
        Xnames = ["lat", "long","year","month","day","dayofyear","weekofyear","dayofweek","weekday","quarter"]
        X = self.dftaxi[Xnames]
        print(X.head())

        # Split off the target (which will be the logarithm of the number of pickups (+1))
        y = self.dftaxi["num_pickups"]
        print(y.head())
        Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
        n_samples = Xtrain.shape[0]
        n_features = Xtrain.shape[1]
        print (Xtrain.shape)
        print(Xtrain.head())

        # Create a Random Forest Regression estimator
        estimator = RandomForestRegressor(n_estimators=20, n_jobs=-1)
        # Define a grid of parameters over which to optimize the random forest
        # We will figure out which number of trees is optimal
        parameters = {"n_estimators": [50],
              "max_features": ["auto"], # ["auto","sqrt","log2"]
              "max_depth": [50]}
        best = self.validation_optimize(estimator, parameters, Xtrain, ytrain, n_folds=5, score_func='mean_squared_error')
        # Fit the best Random Forest and calculate R^2 values for training and test sets
        reg=best.fit(Xtrain, ytrain)
        training_accuracy = reg.score(Xtrain, ytrain)
        test_accuracy = reg.score(Xtest, ytest)
        print ("############# based on standard predict ################")
        print ("R^2 on training data: %0.4f" % (training_accuracy))
        print ("R^2 on test data:     %0.4f" % (test_accuracy))
        modelPred = reg.predict(Xtest)
        meanSquaredError=mean_squared_error(ytest, modelPred, multioutput='raw_values')
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)



