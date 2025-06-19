# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.linear_model.Read the student_score.csv file using pandas.
2. Set the independent variable X as the number of hours studied. Set the dependent variable y as the marks scored.
3. Use train_test_split() to split the data into training and testing sets (e.g., 80% training and 20% testing).
4. Use LinearRegression() from sklearn to train the model using the training data. Predict the scores using the test data.
5. Extract the slope (coef_) and intercept (intercept_) of the best-fit line. 
6. Plot the original data points as a scatter plot and overlay the regression line.

## Program:
```
/*
# Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: Akshaya S K
# Register Number: 212223040011

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
data = pd.read_csv("student_scores.csv")  

# 2. Split into input (X) and output (y)
X = data[['Hours']]   # Feature: Hours studied
y = data['Scores']    # Target: Marks scored

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the linear regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)

# 6. Output the model parameters
print(f"Equation of the line: Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}")

# 7. Plotting the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression - Marks Prediction')
plt.legend()
plt.grid(True)
plt.show()
  
*/
```

## Output:
![image](https://github.com/user-attachments/assets/b7463e95-6701-41a9-b323-0ea4588d7d67)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
