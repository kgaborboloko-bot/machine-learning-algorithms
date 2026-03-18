#ClusteringAnalysis
#QuestionFour
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#load and clean the dataset
df = pd.read_csv("wine_clustering.csv")

#display first few rows
print("First 5 Rows: ")
print(df.head())

#check for missing values
print(df.isnull().sum())

#fill missing numeric values with the column mean
df = df.fillna(df.mean())

#standarsize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

#determine number of clusters (elbow method)
inertia = []  # Within-cluster sum of squares

#cluster numbers from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

#plot the elbow curve
plt.figure(figsize = (8, 5))
plt.plot(range(1, 11), inertia, marker = 'o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()

#apply kmeans clustering
#assuming elbow shows 3 clusters
kmeans = KMeans(n_clusters = 3, random_state = 42)
clusters = kmeans.fit_predict(x_scaled)

#add cluster labels to the dataframe
df['Cluster'] = clusters

score = silhouette_score(x_scaled, clusters)
print(f"Silhouette Score: {score:.3f}")

#visulaize the clusters
plt.figure(figsize = (8, 6))
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c = clusters, cmap = 'viridis', s = 50)
plt.title('K-Means Clustering of Wines')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label = 'Cluster')
plt.show()

