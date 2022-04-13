from gettext import npgettext
from pydoc import importfile
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('Data.csv')

x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

dt=RandomForestRegressor(1000)

x_train,x_test,y_train,y_test=train_test_split(x,y)

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

print(r2_score(y_test,y_pred))