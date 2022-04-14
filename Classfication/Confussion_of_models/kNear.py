import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv('Data.csv')

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Splitting the set
X_train,X_test,y_train,y_test= train_test_split(X,y)

# Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)


# Predictions

predicts=classifier.predict(X_test)

# aprrox 0.9 accuracy
print(confusion_matrix(y_test,predicts))
print(accuracy_score(y_test,predicts))
