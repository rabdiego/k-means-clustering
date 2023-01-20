# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing data
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values

# Implementing the elbow method
wcss = list()
rng = [i for i in range(1, 11)]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Training the model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the results
fig1, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

# Visualizing the elbow method
ax1.set_title('Elbow method', fontsize=26)
ax1.set_xlabel('Number of clusters', fontsize=20)
ax1.set_ylabel('WCSS', fontsize=20)
ax1.plot(rng, wcss)

# Visualizing the clusters
ax2.set_title('Clusters', fontsize=26)
ax2.set_xlabel('Anual Income', fontsize=20)
ax2.set_ylabel('Spending Score', fontsize=20)
ax2.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c='r', label='Cluster 1')
ax2.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c='b', label='Cluster 2')
ax2.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c='g', label='Cluster 3')
ax2.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c='y', label='Cluster 4')
ax2.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], c='m', label='Cluster 5')
ax2.legend(loc=7)
fig1.savefig('plot.png')
