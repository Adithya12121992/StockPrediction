# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:43:06 2020

@author: Adithya & Aradhana
"""

import pandas as pd
import numpy as np

df = pd.read_csv("stockhistoricalProceDataAAPL.csv") # read csv file

rows = df.values.tolist()  # convert dataframe into a list
rows.reverse()

from sklearn.model_selection import train_test_split
x_train = []
y_train = []
x_test = []
y_test = []
X = []
Y = []
for row in rows:
    X.append(int(''.join(row[0].split('-'))))
    Y.append(row[3])
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9,test_size=0.1) # split training and test data

# Convert lists into numpy arrays 
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# reshape the values as we have only one input feature
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

# Linear Regression model
from sklearn.linear_model import LinearRegression 
clf_lr = LinearRegression()
clf_lr.fit(x_train,y_train)
y_pred_lr = clf_lr.predict(x_test)


# Support Vector Machine with a Radial Basis Function as kernel 
from sklearn.svm import SVR
clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf_svr.fit(x_train,y_train)
y_pred_svr = clf_svr.predict(x_test)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(x_train,y_train)
y_pred_rf = clf_rf.predict(x_test)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(x_train,y_train)
y_pred_gb = clf_gb.predict(x_test)

# Quadratic Regression 2
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(x_train, y_train)
y_pred_qd=clfpoly2.predict(x_test)


# KNN Regression
from sklearn.neighbors import KNeighborsRegressor
clfknn = KNeighborsRegressor(n_neighbors=6)
clfknn.fit(x_train, y_train)
y_pred_knn=clfknn.predict(x_test)


import matplotlib.pyplot as plt 

f,(ax1,ax2) = plt.subplots(1,2,figsize=(30,10))

# Linear Regression
ax1.scatter(range(len(y_test)),y_test,label='data')
ax1.plot(range(len(y_test)),y_pred_lr,color='green',label='LR model')
ax1.legend()

# Support Vector Machine
ax2.scatter(range(len(y_test)),y_test,label='data')
ax2.plot(range(len(y_test)),y_pred_svr,color='orange',label='SVM-RBF model')
ax2.legend()

f1,(ax3,ax4) = plt.subplots(1,2,figsize=(30,10))

# Random Forest Regressor
ax3.scatter(range(len(y_test)),y_test,label='data')
ax3.plot(range(len(y_test)),y_pred_rf,color='red',label='RF model')
ax3.legend()

# Gradient Boosting Regressor
ax4.scatter(range(len(y_test)),y_test,label='data')
ax4.plot(range(len(y_test)),y_pred_gb,color='black',label='GB model')
ax4.legend()

f2,(ax5,ax6) = plt.subplots(1,2,figsize=(30,10))
# quardratic poly 2
ax5.scatter(range(len(y_test)),y_test,label='data')
ax5.plot(range(len(y_test)),y_pred_qd,color='blue',label='Quadratic model')
ax5.legend()

# KNN
ax6.scatter(range(len(y_test)),y_test,label='data')
ax6.plot(range(len(y_test)),y_pred_knn,color='black',label='KNN model')
ax6.legend()




print("Accuracy of Linear Regerssion Model:",clf_lr.score(x_test,y_test))
print("Accuracy of SVM-RBF Model:",clf_svr.score(x_test,y_test))
print("Accuracy of Random Forest Model:",clf_rf.score(x_test,y_test))
print("Accuracy of Gradient Boosting Model:",clf_gb.score(x_test,y_test))
print("Accuracy of quadratic model:",clfpoly2.score(x_test,y_test))
print("Accuracy of knn Model:",clfknn.score(x_test,y_test))






