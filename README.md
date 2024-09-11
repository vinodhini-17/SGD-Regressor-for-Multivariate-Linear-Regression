# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To train, initialize theta and iteratively update it using gradient descent.
2.For preprocessing, read and scale data.
3.For modeling, train the linear regression model.
4.To predict data values, scale a new data and predict.
5.Print the prediction.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: vinodhini k 
RegisterNumber:  212223230245
*/
```
~~~~
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
~~~~

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
