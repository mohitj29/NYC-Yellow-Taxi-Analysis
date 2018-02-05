"""
Code containing class and methods for fare tip analysis and taxi locations at different times.
Author : Group 5 , fall 2017
"""
#importing required predefined libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#incorporating specific functions from the libraries
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Class to hold the variables for Tip calculation and taxi location identification
class FareTipPrediction :
    # constructor - loading the dataframe
    def __init__(self, filePath) :
        self.fare = pd.read_csv(filePath)        
        #print(self.fare.head(5))
        
    # Visualization plot based on the analysis 
    def plotGraph (self, xLabel, yLabel, title, groupBy, plotType) :
        fig,ax = plt.subplots(1,figsize = (7,4))
        #checking the type of payment mode
        ax = self.fare.groupby([groupBy]).size().plot(kind=plotType)
        ax.set_xlabel(xLabel, fontsize=12)
        ax.set_ylabel(yLabel, fontsize=12)
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=12)
        plt.show()
        
       
    # method to calculate the tip rate 
    def calculateTipPercentage(self,rec):
        subtotal = rec.fare_amount + rec.mta_tax +rec.extra
        tip = rec.tip_amount / subtotal
        tip_perc = tip * 100 
        return pd.Series({'tip_percentage': tip_perc})
    
    #Pre processing methods 
    # method for cleaning the columns
    def dataCleanup (self , cleanupPaymentType) :
        print("=============================================================")
        print("sum of the missing values in the dataset",self.fare.isnull().sum().sum())
        #print("Missing values in the dataset - columnwise",fare.isnull().sum())       
       
        self.fare.replace(cleanupPaymentType ,inplace = True)
        #checking the missing values
        #print("sum of the missing values in the dataset",self.fare.isnull().sum().sum())
        print("Missing values in the dataset - columnwise",self.fare.isnull().sum())
        print("============================================================")
        
        #drop the columns (based on index) which has more missing values as it will not useful for analysis
        #self.fare.drop(self.fare.columns[[4,14,15,16,17,18,19]], axis=1, inplace=True)
        self.fare.drop(self.fare.columns[[4,14,15,16,18]], axis=1, inplace=True)
        print("after dropping :", self.fare.shape)
        print("============================================================")
   # method to remove the outliers of the features and plot visualization for pickup  and drop off density
    def fareDataAnalysis (self) :                   
        #including card as it relates to the tip amount
        payment_type = (self.fare.payment_type == 'CARD')
        #including the valid fare amount
        fare_amount = ((self.fare.fare_amount >= 3.0) & (self.fare.fare_amount <= 250.0))
        #MTA tax takes the majority value of 0.5
        mta_tax = (self.fare.mta_tax == 0.5)
        tip_amount = ((self.fare.tip_amount >= 0.0) & (self.fare.tip_amount <= 150.0))
        tolls_amount = ((self.fare.tolls_amount >= 0.0) & (self.fare.tolls_amount <= 30.0))
        passenger_count = ((self.fare.passenger_count >= 1.0) & (self.fare.passenger_count <= 6.0))
        trip_distance = ((self.fare.trip_distance > 0.0) & (self.fare.trip_distance <= 50.0))        
        self.fare = self.fare[payment_type & fare_amount & mta_tax & tip_amount & tolls_amount & passenger_count & trip_distance]
        
        pickup_latitude = ((self.fare.pickup_latitude >= 40.459518) & (self.fare.pickup_latitude <= 41.175342))
        pickup_longitude = ((self.fare.pickup_longitude >= -74.361107) & (self.fare.pickup_longitude <= -71.903083))
        dropoff_latitude = ((self.fare.dropoff_latitude >= 40.459518) & (self.fare.dropoff_latitude <= 41.175342))
        dropoff_longitude = ((self.fare.dropoff_longitude >= -74.361107) & (self.fare.dropoff_longitude <= -71.903083))
        self.fare = self.fare[pickup_latitude & pickup_longitude & dropoff_latitude & dropoff_longitude]
        
    
        #visualization on the pickup and drop off locations
        samplesize = 41013
        indices = np.random.choice(self.fare.index, samplesize)
        pickup_xaxis = self.fare.pickup_longitude[indices].values
        pickup_yaxis = self.fare.pickup_latitude[indices].values
        dropoff_xaxis = self.fare.dropoff_longitude[indices].values
        dropoff_yaxis = self.fare.dropoff_latitude[indices].values
        
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(11, 12))
        ax.scatter(pickup_xaxis, pickup_yaxis, s=7, color='blue', alpha=0.7)
        ax.scatter(dropoff_xaxis, dropoff_yaxis, s=5, color='#F5A422', alpha=0.5)
        ax.set_xlim([-74.03, -73.90])
        ax.set_ylim([40.63, 40.85])
        ax.set_xlabel("Longitudinal Coordinates", fontsize = 12)
        ax.set_ylabel("Latitudinal Coordinates", fontsize = 12)
        ax.legend(['Pickup locations','Dropoff locations'],loc='best',title='Group')
        ax.set_title('Map for pickups and drop off locations', fontsize = 20)
        plt.show()
        

        # dropping the payment_type column as only card payment records are segragated 
        self.fare.drop(['payment_type'], axis=1, inplace=True)
        #print(fare.head(5))
        
        #Calculating the tip percentage
        #tip_perc=(tip_amount/fare_amount+mta_tax+extra)*100
        tip_perc_col = 'tip_percentage'
        
        #creating the column and initializing to be nan
        self.fare[tip_perc_col] = np.nan
        # calling the method to calculate the tip percentage and to the new variable
        temp_col = self.fare.apply(self.calculateTipPercentage, axis=1)
        self.fare.update(temp_col)
        temp_col = None
        #print(fare.head(5)) #cansee the new column added to the data frame
        
        #tip percentage cannot be more than 100% so removing the outliers 
        tip_perc = (self.fare[tip_perc_col] <= 100.0)
        self.fare = self.fare[tip_perc]
        
        
    # method to create the model on training data and calculate the accuracy performance metrics on test data
    # 2 models created with multiple linear regression and Random Forest Regression
    #Random Forest Regression Model turned out to be the best model for the prediction of tip percentage
    def modelTipPercentagePrediction (self) :
        # inorder to create the model, let us split the data into training and testing
        #gets a random 80% of the entire set
        train_fare, test_fare = train_test_split(self.fare, test_size=0.2)
        print(train_fare.describe()) #8203
        
        #keeping significant variables
        X_train = train_fare.ix[:,(0,1,6,8,9,10,11)].values
        X_test = train_fare.ix[:,(0,1,6,8,9,10,11)].values
        Y_train = train_fare.ix[:,12].values
        Y_test = train_fare.ix[:,12].values
        
        #scaling the variables for linear regression
        X, Y = scale(X_train),Y_train   
        
        #creating a model using linear regression to predict the tip percentage
        lm = LinearRegression()
        lm.fit(X,Y)
        
        #R squared value calculation
        print("============================================")
        print("R Squared value for linear regression model :",lm.score(X,Y))
        print("Estimated intercept Coefficients :", lm.intercept_)
        print("Estimated Co-efficients :",lm.coef_)
        print("Estimated Number of Co-efficients :",len(lm.coef_))
        lm.predict(X_test)
        print("===========================================")        
        #creating model with random forest to predict the tip percentage
        rf_model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=1234)
        rf_model.fit(X_train, Y_train)
        
        #prediction on the testing data
        predicted_test = rf_model.predict(X_test)
        
        #R squared value calculation for random forest
        test_score = r2_score(Y_test, predicted_test)
        print("R Squared value for random forest regression model :",test_score)
        
        #Root Mean squared error calculation to test the performance for regression model
        rms_LR = np.sqrt(mean_squared_error(Y_test, lm.predict(X_test)))
        print("Root Mean squared Error for Linear regression :",rms_LR)
        
        #Root Mean squared error calculation to test the performance for random forest model
        rms_rf = np.sqrt(mean_squared_error(Y_test, predicted_test))
        print("Root Mean squared Error for Random Forest regression :",rms_rf) #rmse shows the least error and model has a R squared value of 99.9% 
        
        #plot for random forest model prediction to predict the tip percentage
        fig, ax5 = plt.subplots(figsize=(7, 4))
        ax5.scatter(Y_train,predicted_test)
        ax5.set_xlabel("Actuals values",fontsize =12)
        ax5.set_ylabel("Predicted values", fontsize =12)
        ax5.set_title("Actual Tip Percentage Vs Predicted Tip Percentage", fontsize =15)
        plt.show()
        
        #This shows that the model's predicted values are very close with the actual ones. 
        #So the random forest model can be considered as the trustable model to predict the tip percentage 

        
    def getDataFrame(self):
        return self.fare
    
    
    # Logical calculations of area for pickup and dropoff locations  with lat and long coordinates
    def calculate_area(self,rec):
        pickup_xaxis = int((rec.pickup_longitude - self.min_xaxis) / self.step_xaxis)
        pickup_yaxis = int((rec.pickup_latitude - self.min_yaxis) / self.step_yaxis)        
        pickup_area = (pickup_yaxis * self.num_columns) + pickup_xaxis
        
        dropoff_xaxis = int((rec.dropoff_longitude - self.min_xaxis) / self.step_xaxis)
        dropoff_yaxis = int((rec.dropoff_latitude - self.min_yaxis) / self.step_yaxis)
        dropoff_area = (dropoff_yaxis * self.num_columns) + dropoff_xaxis       
        
        return pd.Series({
            'pickup_area' : pickup_area,
            'dropoff_area' : dropoff_area
        })
    
   # assigning the coordinates for area matrix calculation    
    def assignCoordinates(self) :
        self.min_xaxis = -74.361107
        self.min_yaxis = 40.459518
        self.max_xaxis = -71.903083
        self.max_yaxis = 41.175342
        self.step_xaxis = 0.003205  # 270 meters (885 feet) approximately.
        self.step_yaxis = 0.002429  # 270 meters (885 feet) approximately.
        # Number of maximum areas per axis in the 'area matrix'.
        self.max_area_xaxis = 766  # int((max_xaxis - min_xaxis) / step_xaxis)
        self.max_area_yaxis = 294  # int((max_yaxis - min_yaxis) / step_yaxis)
        # Number of columns of the 'area matrix'.
        self.num_columns = self.max_area_xaxis + 1  # max_area_xaxis + 1
        # The area at the top right corner in the 'area matrix'.
        self.max_area = 226264  # (max_area_yaxis * num_columns) + max_area_xaxis
        # Minimum number of pickups or dropoffs per area.
        self.min_area_number = 20
    
    #Removing the outliers and call the calculate area function to do the calculation of area matrix on pickup and drop off location
    def pickAndDropAnalysis(self) :
        #including the valid fare amount
        fare_amount = ((self.fare.fare_amount >= 3.0) & (self.fare.fare_amount <= 250.0))
        passenger_count = ((self.fare.passenger_count >= 1.0) & (self.fare.passenger_count <= 6.0))
        trip_distance = ((self.fare.trip_distance > 0.0) & (self.fare.trip_distance <= 50.0))        
        pickup_latitude = ((self.fare.pickup_latitude >= 40.459518) & (self.fare.pickup_latitude <= 41.175342))
        pickup_longitude = ((self.fare.pickup_longitude >= -74.361107) & (self.fare.pickup_longitude <= -71.903083))
        dropoff_latitude = ((self.fare.dropoff_latitude >= 40.459518) & (self.fare.dropoff_latitude <= 41.175342))
        dropoff_longitude = ((self.fare.dropoff_longitude >= -74.361107) & (self.fare.dropoff_longitude <= -71.903083))
        self.fare = self.fare[pickup_latitude & pickup_longitude & dropoff_latitude
                              & dropoff_longitude & fare_amount & passenger_count & trip_distance]
        self.fare['pickupTime'] = pd.DatetimeIndex(self.fare["pickup_time"]).hour 
        self.fare['dropOffTime'] = pd.DatetimeIndex(self.fare["dropoff_time"]).hour
        print(self.fare.head(5))
        
        self.fare['pickup_area'] = np.nan
        self.fare['dropoff_area'] = np.nan
        # calling the method to calculate the tip percentage and to the new variable
        temp_col = self.fare.apply(self.calculate_area, axis=1)
        self.fare.update(temp_col)
        temp_col = None
        
    #plot for visualizing the pickup area and dropoff area vs hourly trip data    
    def plotPickDropTime (self) :
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(11, 12))
        ax.scatter(self.fare.pickupTime, self.fare.pickup_area, color='black', s = 17, alpha = 0.8)
        ax.scatter(self.fare.dropOffTime, self.fare.dropoff_area, color='orange', s =10, alpha = 0.8)
        ax.set_ylim([30000, 170000])
        ax.set_xlabel("Time in Hours", fontsize = 12)
        ax.set_ylabel("Area Coordinates", fontsize = 12)
        ax.legend(['Pickup locations','Dropoff locations'],loc='best',title='Group')
        ax.set_title('pickups and drop off locations Vs Time', fontsize = 20)
        plt.show()
        
       
        
        
        
        
        
        
    



    
