import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Given data points
data_points = np.array([(2, 3), (5, 6), (8, 7), (1, 4), (2, 2), (6, 7), (3, 4), (8, 6)])

# Number of clusters
num_clusters = 3

# Creating a KMeans instance
kmeans = KMeans(n_clusters=num_clusters)

# Fitting the model to data
kmeans.fit(data_points)

# Getting cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualizing the results
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(data_points[cluster_labels == i, 0], data_points[cluster_labels == i, 1], label=f'cluster{i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200,label='centriod', c='black')
plt.xlabel('X-axis')
plt.ylabel('y-axis')

plt.title('k-means clustering')
plt.legend()
plt.show()
