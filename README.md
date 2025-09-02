# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1: 
Data Collection & Preprocessing Import required libraries (pandas, numpy, matplotlib, sklearn). Load the dataset (student_scores.csv). Separate the independent variable (Hours) and dependent variable (Scores). Split the dataset into training and testing sets.

### Step 2: 
Model Training Initialize the Linear Regression model. Train the model using the training dataset (x_train, y_train).

### Step 3: 
Model Prediction & Visualization Predict scores for the test dataset. Plot the regression line with training data (gradient color for points). Plot the regression line with testing data (compare actual vs predicted values).

### Step 4: 
Model Evaluation Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Display evaluation results to assess model accuracy.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rhudhra phriyamvadha K S
RegisterNumber: 212224040275

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
```

```
df= pd.read_csv('student_scores.csv')
```

```
print("displaying the First 5 Rows")
df.head()
```

```
print("displaying the Last 5 Rows")
df.tail()
```

```
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y
```

```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)
```

```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
```

```
y_pred=regressor.predict(X_test)
y_pred
```

```
y_test
```

```
print("Name: Rhudhra phriyamvadha K S")
print("Reg.No: 212224040275\n")

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```

```
print("Name: Rhudhra phriyamvadha K S")
print("Reg.No: 212224040275\n")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```

```
print("Name: Rhudhra phriyamvadha K S")
print("Reg.No: 212224040275")

mse=mean_squared_error(y_test,y_pred)
print('\nMSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

<img width="240" height="241" alt="image" src="https://github.com/user-attachments/assets/b3d36635-0be8-4fb4-bdb5-e072acf50db1" />
<img width="251" height="247" alt="image" src="https://github.com/user-attachments/assets/902ff211-514c-4c1a-a0ef-679f6ab4cd72" />
<img width="735" height="60" alt="image" src="https://github.com/user-attachments/assets/9cc2ef01-0d46-4b13-8cf3-826e482ea2e6" />
<img width="606" height="67" alt="image" src="https://github.com/user-attachments/assets/71aa6ff0-c882-4d9a-9199-31e9a2f78554" />
<img width="552" height="47" alt="image" src="https://github.com/user-attachments/assets/0f80d4d0-b62a-4051-a859-1732f4ffdf60" />
<img width="688" height="602" alt="image" src="https://github.com/user-attachments/assets/bfe2abd0-afb0-4039-a101-e0c05cd13d8f" />
<img width="682" height="597" alt="image" src="https://github.com/user-attachments/assets/f67f03b3-5976-49cc-80a6-b6fc0d8e68bf" />
<img width="793" height="126" alt="image" src="https://github.com/user-attachments/assets/a7976914-8b60-4e16-a27b-97bd51357900" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
