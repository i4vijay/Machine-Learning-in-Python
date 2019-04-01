import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

# Splitting dataset to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# Model Building
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, Y_train)

# Predicting the Salary
prediction = lm.predict(X_test)
