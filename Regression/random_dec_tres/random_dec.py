from gettext import npgettext
from pydoc import importfile
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor


df=pd.read_csv('Position_Salaries.csv')

x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

dt=RandomForestRegressor(1000)

dt.fit(x,y)
print(dt.predict([[6.5]]))


x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y)
plt.plot(x_grid,dt.predict(x_grid),color='red')
plt.show()