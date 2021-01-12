# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fit() expects 2 args: independent and dependent variables of training set
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_mod = regressor.predict(X_test)                                # regressor.predict() returns a vector of salaries

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')                       # plt.scatter() takes the x and y values of points and plots them
plt.plot(X_train, regressor.predict(X_train), color='blue')      # plt.plot() takes a function and plots it
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')                       # plt.scatter() takes the x and y values of points and plots them
plt.plot(X_train, regressor.predict(X_train), color='blue')      # plt.plot() takes a function and plots it
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 

# Finding a single prediction
# regressor.predict() takes a 2D Array 
# Scalar: 12
# 1D Array: [12]
# 2D Array: [[12]]
# regressor.predict returns a 1D array
print(regressor.predict([[12]]))

# Finding b0 and b1 in regression equation
# regressor.coef_ returns a 1D array
# regressor.intercept_ returns an integer 
# .coef_ and .intercept_ are attributes of the LinearRegression object 
b0 = regressor.intercept_
b1 = regressor.coef_
print('The regression equation is y = ' + str(b0) + ' + ' + str(b1[0]) + 'x')