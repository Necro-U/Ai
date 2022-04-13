from sklearn.preprocessing import OneHotEncoder,LabelEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading values
df=pd.read_csv('Data.csv')

# Assigning values. first column is not added because level column is representing it
X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values

x_train,x_test,y_trein,y_test=train_test_split(X,Y)

# Fitting
poly_reg=PolynomialFeatures(4)
X_poly=poly_reg.fit_transform(x_train)

lr2=LinearRegression()
lr2.fit(X_poly,y_trein)
y_pred=lr2.predict(x_test)
print(r2_score(y_test,y_pred))