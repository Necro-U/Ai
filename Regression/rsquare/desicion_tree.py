from string import printable
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('Data.csv')
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y)


regressor=DecisionTreeRegressor()



regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(r2_score(y_test,y_pred))