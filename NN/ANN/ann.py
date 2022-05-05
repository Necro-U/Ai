from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv("Churn_Modelling.csv")

X=df.iloc[:,3:-1].values
y=df.iloc[:,-1].values

le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

# onehotencoderin yanındaki sayı kaçıncı column olduğunu gösteriyor!
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')

X=np.array(ct.fit_transform(X))

print(X[:,:6])
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#initiliaze ANN
ann=tf.keras.models.Sequential()

# adding layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# fitting 
#binary_crossentropy is for binary cases we have 2 different cases 
ann.compile(optimizer='adam', loss ='binary_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,batch_size=32,epochs=100)

## applying to one person
# man=[[1.0,0.0,0.0,600,1,40,3,60000,2,1,1,50000]]
# man=sc.transform(man)
# print(ann.predict(man))

y_pred=ann.predict(x_test)
y_pred= y_pred>0.5
print(confusion_matrix(y_test,y_pred))