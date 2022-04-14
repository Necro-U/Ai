# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn. model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix , accuracy_score

# df = pd.read_csv('Social_Network_Ads.csv')

# X=df.iloc[:,:-1].values
# Y=df.iloc[:,-1].values

# regressor=LogisticRegression()

# x_train , x_test , y_train ,y_test =train_test_split(X,Y,test_size=0.3)

# y_test=y_test.reshape((len(y_test),1))
# scaler_x=StandardScaler()
# x_train=scaler_x.fit_transform(x_train)
# x_test=scaler_x.transform(x_test)
# print(x_train.shape)
# print(x_test.shape)
# print(scaler_x.inverse_transform(x_test).shape)
# print(y_test.shape)



# regressor.fit(x_train,y_train)

# predictions=regressor.predict(x_test)
# ac_score=accuracy_score(y_test,predictions)
# cm=confusion_matrix(y_test,predictions)
# print(cm,ac_score)
# # plt.scatter(scaler_x.inverse_transform(x_test),y_test)
# # plt.plot(scaler_x.inverse_transform(x_test),predictions)
# # plt.show()





# CODS FROM İNTERNET


# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
