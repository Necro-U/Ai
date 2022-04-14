import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score


# Load Data
df = pd.read_csv('Data.csv')

# Seperate data
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# Seperate data to train-test
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Future Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Create estimator object
classifier=SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)



# Avaluate predictions
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))