import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv('Data.csv')


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values



X_train,X_test,y_train,y_test=train_test_split(X,y)



# furture scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


predicts=classifier.predict(X_test)


print(confusion_matrix(y_test,predicts))
print(accuracy_score(y_test,predicts))