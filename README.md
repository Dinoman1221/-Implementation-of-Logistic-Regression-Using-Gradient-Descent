# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Data preprocessing:
3. Cleanse data,handle missing values,encode categorical variables.
4. Model Training:Fit logistic regression model on preprocessed data.
5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6. Prediction: Predict placement status for new student data using trained model.
7. End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Ashish S
RegisterNumber:  24900566
*/
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
print(data.head())
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print(data1)
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
https://github.com/user-attachments/assets/d4059d91-0c38-4adc-a963-dde1414be9f5



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

