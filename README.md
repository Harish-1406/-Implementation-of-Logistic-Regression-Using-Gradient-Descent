# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages

2.Read the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary and predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HARISH P K
RegisterNumber:  212224040104
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop(['sl_no','salary'],axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
x=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew) 
```


## Output:
![image](https://github.com/user-attachments/assets/156b5eec-2ac3-4d99-87d2-cca5074dde99)
![image](https://github.com/user-attachments/assets/ff7c1a9c-70f1-44b6-b6ed-ccd4d260913a)
![image](https://github.com/user-attachments/assets/2cfe65f1-65fb-4e32-8515-67cdb0d07f24)
![image](https://github.com/user-attachments/assets/bf7f6a3a-f26d-4638-b8e6-bfc215e9c550)
![image](https://github.com/user-attachments/assets/e736dbea-daab-4a2e-8abd-94a4fc2efaee)
![image](https://github.com/user-attachments/assets/6f469707-3836-4343-8113-d8ccc4ba2c51)
![image](https://github.com/user-attachments/assets/ae7574ef-6c01-4cda-a4d0-214f95a9d79d)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

