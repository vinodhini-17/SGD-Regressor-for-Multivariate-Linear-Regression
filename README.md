# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load California housing data, select features and targets, and split into training and testing sets.
2.Scale both X (features) and Y (targets) using StandardScaler.
3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4.Predict on test data, inverse transform the results, and calculate the mean squared error.
````
## Program:

`````
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: vinodhini k
RegisterNumber:  212223230245
``````
`````
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
`````
![image](https://github.com/user-attachments/assets/4641f46e-737d-40b4-811a-03cfc3187fd9)
``
df.info()
``
![image](https://github.com/user-attachments/assets/bab93aab-419c-46e3-9be2-a35fd87559c8)
````
X=df.drop(columns=['AveOccup','target'])
X.info()
````
![image](https://github.com/user-attachments/assets/7d5d1338-b564-44a6-8dca-de909cb2778b)
````
Y=df[['AveOccup','target']]
Y.info()

![image](https://github.com/user-attachments/assets/e03d08a7-6e3a-48a7-84f4-1d146692dca2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
print(X_train)
```````
![image](https://github.com/user-attachments/assets/e26c6a8e-3d8a-485c-bc8f-b08f2d772077)
# Initialize the SGDRegressor
`````
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
```````
# Use MultiOutputRegressor to handle multiple output variables
``````
multi_output_sgd = MultiOutputRegressor(sgd)
``````````
# Train the model
`````
multi_output_sgd.fit(X_train, Y_train)
````````
# Predict on the test data
```````
Y_pred = multi_output_sgd.predict(X_test)
```````
# Initialize the SGDRegressor
````````
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
`````````
# Use MultiOutputRegressor to handle multiple output variables
```````
multi_output_sgd = MultiOutputRegressor(sgd)
```````````
# Train the model
``````
multi_output_sgd.fit(X_train, Y_train)
```````
# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
````
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
````

# Evaluate the model using Mean Squared Error
````
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
````````

# Optionally, print some predictions
````
print("\nPredictions:\n", Y_pred[:5])
````
![image](https://github.com/user-attachments/assets/7ecfd652-de53-4e89-9def-c5b705cbde25)





## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
