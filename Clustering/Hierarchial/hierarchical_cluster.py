import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

df = pd.read_csv('Mall_Customers.csv')

X=df.iloc[:,[3,4]].values


# plotting dendegram
# dendogram = sch .dendrogram(sch.linkage(X,method='ward'))
# plt.title('dendogram')
# plt.xlabel('Custumers')
# plt.ylabel('Distances of Custumers')
# plt.show()

hc=AgglomerativeClustering(5,linkage='ward')
y_hc=hc.fit_predict(X)

# plotting results
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],color='red',s=100,label='1. class')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],color='blue',s=100,label='2. class')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],color='green',s=100,label='3. class')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],color='gray',s=100,label='4. class')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],color='black',s=100,label='5. class')
# plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1],s=100,c='cyan',label="centers")
plt.legend()
plt.show()