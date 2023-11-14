import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, data: pd.DataFrame):
        """
        Note: `data` MUST be already standardized, so it has mean=0, std=1.

        Example usage:
        Assume 'data' is a DataFrame with features in columns
        and you want to create 3 clusters (k=3)
        labels = k_means(data, k=3)

        You can then add the labels to your DataFrame:
        data['Cluster'] = labels
        """
        self.k = k
        self.data = data
        self.centroids = self.data.sample(self.k).values


    def train(self, max_epochs = 100):
        for epoch in range(max_epochs):
            # Assignment step
            distances = np.linalg.norm(self.data.values[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids step
            new_centroids = np.array([self.data[labels == j].mean(axis=0) for j in range(self.k)])

            # Check for convergence
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids
        
        inertia = np.sum((self.data.values - self.centroids[labels]) ** 2)

        return labels, inertia, epoch


    def predict(self, X):
        distances = np.linalg.norm(X.values[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        return labels
