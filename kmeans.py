from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
file_path = 'Universal Online Judge_联合权值.csv'
data0 = pd.read_csv(file_path)

for i in range(1,11):
    #Choose all cols where `index` = 1
    data = data0.loc[data0['index'] == i]

    # Selecting numerical columns for clustering
    numerical_cols = ['结果', '用时', '内存', '文件大小', 'Score', 'time', 'mem']
    data_numerical = data[numerical_cols]

    # Handling missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_numerical)

    # Standardizing the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # Performing KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Reducing data to 2 dimensions using PCA for visualization
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_scaled)

    # Assuming 'data_scaled' is your standardized dataset and 'kmeans' is your fitted KMeans model

    # Calculate the distance of each point to its centroid
    centroids = kmeans.cluster_centers_
    distances = np.sqrt(((data_scaled - centroids[kmeans.labels_]) ** 2).sum(axis=1))

    # Determine the threshold for being "far away"
    # Here, using 2 standard deviations from the mean distance as the threshold
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 1.7 * std_distance

    # Mark points that are far away
    far_away = distances > threshold

    #print(data_reduced[far_away, 0], data_reduced[far_away, 1])

    # Update the plot to mark far away points in red
    plt.figure(figsize=(10, 6))
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.5)
    #plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=~far_away, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.5, label='Close to centroid')
    plt.scatter(data_reduced[far_away, 0], data_reduced[far_away, 1], color='red', marker='o', edgecolor='k', s=50, alpha=0.5, label='Anomoly')
    plt.title(f'KMeans Clustering Plot with Anomaly Points Marked (index={i})')
    plt.xlabel('PCA Index 1')
    plt.ylabel('PCA Index 2')
    plt.colorbar(label='Cluster')
    plt.legend()
    plt.show()