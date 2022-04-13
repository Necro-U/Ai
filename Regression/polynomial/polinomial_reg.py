from sklearn.preprocessing import OneHotEncoder,LabelEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading values
df=pd.read_csv('Position_Salaries.csv')

# Assigning values. first column is not added because level column is representing it
X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values

# One hot Encoding
# ct=ColumnTransformer(transformers=['encoder',OneHotEncoder(),[0]],remainder='passthrough')
# X=ct.fit_transform(X)

# Fitting
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
lr=LinearRegression()
lr.fit(x_train,y_train)


predictions=lr.predict(x_test)

concate=np.concatenate((predictions.reshape((len(predictions),1)),y_test.reshape((len(y_test),1))),axis=1)
print(concate)

poly_reg=PolynomialFeatures(4)
X_poly=poly_reg.fit_transform(X)

lr2=LinearRegression()
lr2.fit(X_poly,Y)


import matplotlib.pyplot as plt

plt.scatter(X,Y)
plt.plot(X,lr.predict(X),color='red')
plt.plot(X,lr2.predict(X_poly),color='green')
plt.xlabel('level')
plt.ylabel('salaries')
plt.show()





