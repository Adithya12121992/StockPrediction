# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:54:28 2020

@author: Adithya & Aradhana
"""
def senti_write(value):
    text=value.split(":")
    if float(text[1])>0:
        text[1]="predicted value is Bullish : This stock price is rising"
    elif float(text[1])<0:
        text[1]="predicted value is Bearish : This stock price is falling"
    else:
        text[1]="predict value is the Neutral : The stock price remains the same"
    print(text[0]+"\n"+text[1])

import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import os as os
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv(os.path.join('stockhistoricalProceDataAAPL1.csv'),delimiter=',',usecols=['Date','open_price','close_price','high_price','low_price','sentiment'])
df = df.sort_values('Date')# Sort DataFrame by date
print(df.head())# Double check the result
df = df[['sentiment']]
#print(df.head())

# A variable for predicting 'n' days out into the future
forecast_out =  1#'n=30' days
#Create another column (the target ) shifted 'n' units up
df['Prediction'] = df[['sentiment']].shift(-forecast_out)
#print the new data set
#print(df.tail())
### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'],1))
#Remove the last '30' rows
X = X[:-forecast_out]
#print(X)
### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array 
y = np.array(df['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
#print(y)
# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Create and train the Support Vector Machine (Regressor) 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)
# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)



# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)
# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
#print(x_forecast)

clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(x_train,y_train)
rf_confidence = clf_rf.score(x_test, y_test)
print("random forest confidence:   ",rf_confidence)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(x_train,y_train)
gb_confidence = clf_gb.score(x_test, y_test)
print("gradient boosting confidence:   ",gb_confidence)

# Quadratic Regression 2
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(x_train, y_train)
poly2_confidence = clf_gb.score(x_test, y_test)
print("quadratic regression confidence:   ",poly2_confidence)

from sklearn.neighbors import KNeighborsRegressor
clfknn = KNeighborsRegressor(n_neighbors=6)
clfknn.fit(x_train, y_train)
knn_confidence = clfknn.score(x_test, y_test)
print("knn confidence:   ",knn_confidence)


print("\n\n TRADING ADVICE BASED ON SENTIMENT\n\n")
# Print linear regression model predictions for the next '30' days
lr_prediction = lr.predict(x_forecast)
senti_write("lr prediction :"+str(lr_prediction[0]))
#print("lr prediction")
#print(lr_prediction)# Print support vector regressor model predictions for the next '30' days
svm_prediction = svr_rbf.predict(x_forecast)
#print("svm prediction")
#print(svm_prediction)
senti_write("svm prediction :"+str(svm_prediction[0]))
rf_prediction=clf_rf.predict(x_forecast)
#print("rf prediction")
#print(rf_prediction)
senti_write("rf prediction:"+str(rf_prediction[0]))
gb_prediction=clf_gb.predict(x_forecast)
#print("gb prediction")
#print(gb prediction)
senti_write("rf prediction:"+str(gb_prediction[0]))
poly2_prediction=clfpoly2.predict(x_forecast)
#print("poly  prediction")
#print(poly2_prediction)
senti_write("poly  prediction:"+str(poly2_prediction[0]))
knn_prediction=clfknn.predict(x_forecast)
#print("KNN prediction")
#print(knn_prediction)
senti_write("KNN prediction:"+str(knn_prediction[0]))