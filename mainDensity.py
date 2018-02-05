"""
Main code containing method calls and class instances
Author: Group 5, Fall 2017
"""

# importing the analysis part from the Taxi density and pickup analysis file
from all_model import TaxiDensity
#import density_time
# loading the data into the workspace
NYTaxi = TaxiDensity('./data_density.csv')
print("Welcome to NYC Taxi Data Analysis")

#Method call to preprocess the data
print("Data is being preprocessed to remove outliers, null values and out of range data")
NYTaxi.dataPreProcessing()

#Display the processed data
NYTaxi.dataDisplay()

#Method call to visualize the data in histogram and heat maps
print(" VISUAL REPRESENTATION OF THE DATA USING HISTOGRAM AND HEAT MAP:")
NYTaxi.dataHistogram()
NYTaxi.DensityGraph()
NYTaxi.DensityHeatMapGraph()

#Method call to various regression models
print(" Data analysis is done using Poisson, Negative Binomial and Random Forest Regression")
print("Response Variable : Number of Pickups")
print("Predictors considered: Latitude, longitude and Date fields for the taxi")
NYTaxi.RegressionModel()
#NYTaxi.RandomForest()
print("From analysis and based on the RMSE values, poisson regression seems to be the best fit for the model")

## Analysis of taxi density based on Time
NYTaxiTime=TaxiDensity('./2015.csv')
print("=================================================")
print("=================================================")
print("analysis of taxi density based on time considerations...")
print("=========================================================")
#NYTaxiTime.dataPreProcessing()
#NYTaxiTime.dataDisplay()


#code for time based pickup density analysis
import density_time


#### Time series analysis and graphs
print(" TIME SERIES ANALYSIS ON DATA")
import time_series_and_visualize_calling_methods


#Prediction based on location of taxi.
print("\n \n \n \n")
print(" PICK UP DROP OFF LOCATION PREDICTION")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# importing the analysis part from the FareTipPredictionClass file
from FareTipPredictionClass import FareTipPrediction

# loading the data into the workspace
Pred = FareTipPrediction('./final.csv')

cleanup_paymentType = {"payment_type": {"CRD" : "CARD", "CREDIT" : "CARD", "Cre" : "CARD", 1 : "CARD","Credit" : "CARD","1":"CARD" ,
             "CSH" : "CASH", "CAS" : "CASH" ,2 : "CASH", "Cash" : "CASH"}}
    
# calling the datacleanup method
Pred.dataCleanup(cleanup_paymentType)
#calculation of the area matrix for pickup area and dropoff area
Pred.assignCoordinates()
#Removing the outliers and call the calculate area function to do the calculation of area matrix on pickup and drop off location
Pred.pickAndDropAnalysis()
#plot for visualizing the pickup area and dropoff area vs hourly trip data 
Pred.plotPickDropTime()

###########
print(" FARE TIP ANALYSIS")
print("##########################################")
print("++++++++++++++++++++++++++++++++++++++++++")
# calling the datacleanup method
from FareTipPredictionClass1 import FareTipPrediction
fareTip = FareTipPrediction('./final.csv')
fareTip.dataCleanup(cleanup_paymentType)
# Visualization for the MTA TAX Vs Number of trips
fareTip.plotGraph('MTA Tax','Number of Trips','Plot for MTA TAX', 'mta_tax','bar')
# Visualization for the Passenger count Vs Number of trips
fareTip.plotGraph('Passenger count','Number of Trips','Plot for Passenger Count', 'passenger_count','bar')
# Visualization for the Payment Type Vs Number of trips
fareTip.plotGraph('Payment Type','Number of Trips','Plot for Payment Mode', 'payment_type','bar')
# calling the function for outlier removal and visualization on the density of pick up and drop off
fareTip.fareDataAnalysis()
# calling the function for model creation for the prediction of the tip percentage
fareTip.modelTipPercentagePrediction()
# Distribution of the tip percentage over transcation
fareTip.plotGraph('Tip (%)','Number of transcations', 'Distribution of Tip (%) transactions', 'tip_percentage', 'line')

#"Tip_percentage" showed that 60% of all transactions did not give tip 
# A second tip at 40% corresponds to the usual NYC customary gratuity rate which fluctuates between 10% and 40%

print(" ANALYIS SUMMARY MENTIONED IN REPORT!")
print("thank you")

