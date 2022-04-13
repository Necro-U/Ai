# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.svm import SVR
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import StandardScaler


# df=pd.read_csv('Position_Salaries.csv')


# x=df.iloc[:,1:-1].values
# y=df.iloc[:,-1].values
# y=y.reshape(len(y),1)

# print(x,y)

# sc=StandardScaler()
# sc2=StandardScaler()
# x=sc.fit_transform(x)
# y=sc2.fit_transform(y)

# svr_model=SVR()
# svr_model.fit(x,y)

# pred=sc2.inverse_transform(svr_model.predict(sc.transform(x)))

# plt.scatter(sc.inverse_transform(x),sc2.inverse_transform(y))
# plt.plot(sc.inverse_transform(x),pred)
# plt.xlabel('salaries')
# plt.ylabel('levels')
# plt.show()


# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()