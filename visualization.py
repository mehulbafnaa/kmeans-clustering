import matplotlib.pyplot as plt
import numpy as np
from typing import List
from kmeans import Centroid, Cluster

def plot_clusters(points: np.ndarray, clusters: List[Cluster], centroids: List[Centroid], title: str):
    """
    Visualize the clusters and centroids in a 2D scatter plot.

    Args:
        points (np.ndarray): The input data points.
        clusters (List[Cluster]): Cluster assignments for each point.
        centroids (List[Centroid]): List of centroid coordinates.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 7))
    unique_clusters = set(clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_points = points[np.array(clusters) == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], alpha=0.5)
    
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], c='red', marker='x', s=200, linewidths=3)
    
    plt.title(title)
    plt.show()

def plot_elbow(k_values: List[int], inertias: List[float]):
    """
    Create an elbow plot to help determine the optimal number of clusters.

    Args:
        k_values (List[int]): List of k values tested.
        inertias (List[float]): Corresponding inertia values for each k.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()