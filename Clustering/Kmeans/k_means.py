from cProfile import label
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.cluster import KMeans



df= pd.read_csv('Mall_Customers.csv')


X=df.iloc[:,[3,4]].values



# label encoding




# spliting data to train test
X_train,X_test=train_test_split(X)


# deciding how many class variable
wcss=[]
for i in range(1,11):
    cluster=KMeans(i,random_state=42)
    cluster.fit(X_train)
    wcss.append(cluster.inertia_)

# plotting all wcss scores
# plt.plot(range(1,11),wcss)
# plt.show()

cluster_2=KMeans(5)
y_means=cluster_2.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1],color='red',s=100,label='1. class')
plt.scatter(X[y_means==1,0],X[y_means==1,1],color='blue',s=100,label='2. class')
plt.scatter(X[y_means==2,0],X[y_means==2,1],color='green',s=100,label='3. class')
plt.scatter(X[y_means==3,0],X[y_means==3,1],color='gray',s=100,label='4. class')
plt.scatter(X[y_means==4,0],X[y_means==4,1],color='black',s=100,label='5. class')
plt.scatter(cluster_2.cluster_centers_[:,0],cluster_2.cluster_centers_[:,1],s=100,c='cyan',label="centers")
plt.legend()
plt.show()


# # K-Means Clustering

# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('Mall_Customers.csv')
# X = dataset.iloc[:, [3, 4]].values

# # Using the elbow method to find the optimal number of clusters
# from sklearn.cluster import KMeans
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# # Training the K-Means model on the dataset
# kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(X)

# # Visualising the clusters
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()