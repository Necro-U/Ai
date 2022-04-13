import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


df=pd.read_csv('Data.csv')


x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
y=y.reshape(len(y),1)

print(x,y)



sc=StandardScaler()
sc2=StandardScaler()
x=sc.fit_transform(x)
y=sc2.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y)

svr_model=SVR()
svr_model.fit(x_train,y_train)



y_pred=sc2.inverse_transform(svr_model.predict(sc.transform(x_test)))

print(r2_score(y_test,y_pred))