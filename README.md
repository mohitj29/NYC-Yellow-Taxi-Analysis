#                               NYC-Yellow-Taxi-Analysis


![image](https://user-images.githubusercontent.com/27119316/35784609-98e47d48-09e7-11e8-9520-bf627e7c8ef3.png)


### INTRODUCTION

Predictive analytics never fails to amaze us by its incredible usage. It not only helps us what’s going on but also what is likely to happen in the future. When our team started discussing about what problem would be interesting to analyze for PMDS term project we came across different ideas. When we had to choose what was most interesting of all, we voted for NYC taxi prediction problem. As we all have that love for NYC we were really intrigued by the analytics that is going beyond planning of NYC taxis. The following sections will explain the problem we are trying to address, the approach we will be taking and recommendations we offer from results.

### DATASET

NYC open data has made lot of resources available for public. We downloaded our data by writing SQL queries in Google BigQuery. Considering the resource constrains we couldn’t use the entire dataset which was close to 10 GB. Instead we decided to do a cluster sampling to get equal number of observations from each group where groups are years from 2009 to 2016. We have 15000 records for each year from 2009 to 2016. All our analysis is based on TLC data but our data was aggregated across different dimensions based on requirements. Number of pickups field was not present in original dataset and we successfully calculated it on a given location on a given day (and time) by grouping through SQL queries. Also we could download only 15000 observations for a year as beyond that limit had to be paid. The below queries were repeated for years from 2009 to 2016.


####	Number of pickups (scaled)


![image](https://user-images.githubusercontent.com/27119316/35784641-fa9d7756-09e7-11e8-8657-10c09f7e3ea9.png)


We can see that data is positively skewed or skewed towards right


####	Pickup density in NYC (Maximum number of pickups)

![image](https://user-images.githubusercontent.com/27119316/35784644-ffe702c2-09e7-11e8-9c89-670a0287c4fe.png)


All the dots represent significant pickup numbers. The crowded points represent areas where maximum number of pickups have repeated over years.  Those areas needs to be concentrated to improve business.


#### 	Heat map of Quarter and Weekday Vs Number of Pickups


![image](https://user-images.githubusercontent.com/27119316/35784708-86f4742a-09e8-11e8-81f3-1add6457c289.png)


Since the data is aggregated from 2009 to 2016 the total number of pickups are represented as huge numbers. Q4 Mondays are least whereas Q3 Thursday recorded highest number of pickups.


#### 	Heat map of Month and Year Vs Number of Pickups


![image](https://user-images.githubusercontent.com/27119316/35784729-b7c0af74-09e8-11e8-988f-c9b6d9638e65.png)


Since the data is aggregated from 2009 to 2016 the total number of pickups are represented as huge numbers. May 2015 is least whereas 2015 August recorded highest number of pickups.

#### 	Plot for Passenger Count Vs Number of trips


![image](https://user-images.githubusercontent.com/27119316/35784750-01ba1f02-09e9-11e8-9bae-485ad1b791a6.png)


A usual trip has 1 to 6 passengers. Here the passenger count seems to be one in maximum number of trips. Rest of them (0) can be discarded as a data error

#### 	Plot for Payment Mode Vs Number of trips


![image](https://user-images.githubusercontent.com/27119316/35784771-3355abb2-09e9-11e8-8655-42c3455a5a74.png)


The most common payment from the sample collected turned out to be cash as per the above analysis. However, as per data dictionaries provided by NYC Taxi and Limousine Commission, cash tips are not included in the dataset. So, we are considering the CARD payment mode for our analysis.


#### 	Plot for the distribution of tip percentage over the trip fare transactions


![image](https://user-images.githubusercontent.com/27119316/35784787-6349977a-09e9-11e8-8775-a2eb6de56e8e.png)


#### 	PICKUP DENSITY WITH TIME FOR 2015

Considering the count nature of DV (number of pickups) we chose Poisson regression and Negative Binomial Regression. We didn’t go for widely used multiple linear regression as it could give negative values which is not applicable for count data.

SUMMARY


![image](https://user-images.githubusercontent.com/27119316/35784844-ecc60358-09e9-11e8-9742-0d6b8ce5472f.png)


![image](https://user-images.githubusercontent.com/27119316/35784848-f28e96b0-09e9-11e8-842f-1578491e07be.png)


#### Identify pickups which originate and end at same point and Taxi planning can be made effective by making taxis hover around those locations at the given time (hourly basis)

To achieve the objective, we have analyzed the NYC YELLOW TAXI TRIP data features focusing on the pickup and drop off location coordinates, pickup time, drop time and performed a logical calculation of area matrix to get the area coordinates from the pickup and drop off latitudes and longitudes. The below plot shows the pictorial representation of taxi pickup and drop off’s happening in the same location on hourly basis.


![image](https://user-images.githubusercontent.com/27119316/35784860-14275154-09ea-11e8-8cda-2151bec88573.png)

In the plot, the blue dot represents the pickups’ and the orange spots the drop offs on an hourly basis. The points are mostly overlapping which means that the taxi drop off and pickup is happening at the same location and at the same time.

####  Time series Analysis –Detecting Trend and Seasonality in Taxi Requests


![image](https://user-images.githubusercontent.com/27119316/35784880-39ca4cea-09ea-11e8-8550-07f750657640.png)


We have applied time series analysis for a particular area in Manhattan (described in the map).

Summary of the time series of number of pickups from 2009-2016 years:


![image](https://user-images.githubusercontent.com/27119316/35784891-405e0060-09ea-11e8-87f5-fd73add1773c.png)


The above graph is the time series plot starting from January 2009 - June 2016. The number of pickups around the Times Square location was significantly more in 2009, which reached to its peak between 2012-2013 years. After 2013 year, the number of pickups drastically reduces and this trend follows in the upcoming years. 

In the 2014-2016, Uber and Lyft came into the taxi business and figures show that how Uber and Lyft impacted the business of NYC yellow taxi. This could be a one reason for decrease in the number of pickups of NYC taxi from 2013.


![image](https://user-images.githubusercontent.com/27119316/35784916-650c978c-09ea-11e8-8ec3-41ce2ec751b5.png)


![image](https://user-images.githubusercontent.com/27119316/35784931-7eadd430-09ea-11e8-88ba-e30ab2705fef.png)


Source: http://www.businessofapps.com/data/uber-statistics/


### AUTO-CORRELATION


![image](https://user-images.githubusercontent.com/27119316/35784950-97acd67a-09ea-11e8-9a7f-aa8407432520.png)


The above plot is the Autocorrelation Function (ACF), of the data.
The plot shows lag values along the x-axis and correlation on the y-axis between -1 and 1 for negatively and positively correlated lags respectively.
The blue line above the dotted grey line indicate statistical significance.

### Decomposition:


![image](https://user-images.githubusercontent.com/27119316/35784972-c214af46-09ea-11e8-9cc8-5fcf4accf3c0.png)


The decomposition plot explains the observed series, trend line, the seasonal pattern, and random part of the series.The above decomposition plot shows, more influence of trend in the time series, there is decreasing trend in the number of pickups from 2009-2016. 


### Duration of Ride:


![image](https://user-images.githubusercontent.com/27119316/35784989-e912baf2-09ea-11e8-9ae8-ab2ea8eec256.png)


The above box plot represents, the total duration of ride taken by individuals from 2009-2016. As we can infer from the box plot, that most of the rides duration is under 50 minutes. There is lots of outliers present in the box plot, which we can consider as an effect of rush hours.


![image](https://user-images.githubusercontent.com/27119316/35784990-ecf35424-09ea-11e8-9b6e-21b3dc05cd64.png)


## RECOMMENDATIONS


•	Strategic business planning – Taxi companies can effectively position taxis based on demand during peak hours and also dispatch optimal number of taxis accordingly using the predictive models

•	Traffic Management – City planners can use this model to understand traffic at a specific location on a given day for efficient traffic management. Concentrate on areas shown as highly dense by the visualizations 

•	Seasonal Factors – Taxi planners can make use of heat maps and time series analysis findings to identify seasons when the taxi requests will be high and formulate strategies

•	Ridership behavioral analysis – Identifying possible patterns in passenger tipping behavior using the predictive model  to provide customized and improved services 
