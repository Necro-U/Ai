import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv('Social_Network_Ads.csv')

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test= train_test_split(X,y)


classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)


# Predictions

predicts=classifier.predict(X_test)
print(predicts)