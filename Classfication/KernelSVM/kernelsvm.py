import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('Social_Network_Ads.csv')

# Seperate data
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# Seperate data to train-test
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Future Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Kernels
kernels=['rbf','linear','poly','sigmoid']

# For each kernel find accuracy
for kernel in kernels:
    print(f"SVM kernel : {kernel} ")

    # Create estimator object

    classifier=SVC(kernel=kernel)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)



    # Avaluate predictions
    print(f"{kernel}'s conf matrix : \n{confusion_matrix(y_test,y_pred)}")
    print(f"{kernel}'s accuracy : {accuracy_score(y_test,y_pred)}")

print("Kernel Finished...")