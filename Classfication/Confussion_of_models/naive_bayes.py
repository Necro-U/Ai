import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('Data.csv')

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


X_train,X_test,y_train,y_test=train_test_split(X,y)


#future scaling

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#creating Gaussian NB model
gnb=GaussianNB()
print("Gaussian NB is training")
gnb.fit(X_train,y_train)
# making predictions
y_pred_gnb=gnb.predict(X_test)
# avaluating
print(confusion_matrix(y_test,y_pred_gnb))
print(accuracy_score(y_test,y_pred_gnb))


#creating Bernoulli NB model
gnb=BernoulliNB()
print("Bernoulli NB is training")
gnb.fit(X_train,y_train)
# making predictions
y_pred_gnb=gnb.predict(X_test)
# avaluating
print(confusion_matrix(y_test,y_pred_gnb))
print(accuracy_score(y_test,y_pred_gnb))


