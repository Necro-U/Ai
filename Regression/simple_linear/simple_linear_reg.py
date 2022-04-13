from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('Salary_Data.csv')


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


regressor = LinearRegression()

regressor.fit(x_train,y_train)


predicts=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.show()