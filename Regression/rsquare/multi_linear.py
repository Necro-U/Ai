import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("Data.csv")

x= df.iloc[:,:-1].values
y= df.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)

print(r2_score(y_test,y_pred))