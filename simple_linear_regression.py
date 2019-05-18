#Simple Linear Regression

#Date PreProcessing
#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Taking care of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy= 'mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Splitting Data into sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
