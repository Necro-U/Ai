from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('Position_Salaries.csv')
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

regressor=DecisionTreeRegressor()

regressor.fit(x,y)

print(regressor.predict([[6.5]]))


x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))

plt.scatter(x,y)
plt.plot(x_grid,regressor.predict(x_grid),color='red')
plt.show()