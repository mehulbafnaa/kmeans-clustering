import numpy as np
from typing import List
from kmeans import kmeans, euclidean_distance
from sklearn.cluster import KMeans as SklearnKMeans
import time

def compare_kmeans(points: np.ndarray, k: int, max_iters: int = 100):
    """
    Compare custom K-means implementation with sklearn's K-means.

    This function runs both implementations on the same dataset and compares
    their execution time, inertia, cluster assignments, and cluster sizes.

    Args:
        points (np.ndarray): Input data points to cluster.
        k (int): Number of clusters.
        max_iters (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: Custom clusters, custom centroids, sklearn clusters, sklearn centroids.
    """
    # Custom K-means
    start_time = time.time()
    custom_clusters, custom_centroids = kmeans(points, k, max_iters=max_iters, init_method='kmeans++', random_state=42)
    custom_time = time.time() - start_time
    
    # Sklearn K-means
    sklearn_kmeans = SklearnKMeans(n_clusters=k, max_iter=max_iters, init='k-means++', n_init=1, random_state=42)
    start_time = time.time()
    sklearn_clusters = sklearn_kmeans.fit_predict(points)
    sklearn_time = time.time() - start_time
    
    # Calculate inertias
    custom_inertia = sum(min(euclidean_distance(point, centroid)**2 for centroid in custom_centroids) for point in points)
    
    # Print results
    print(f"Custom K-means execution time: {custom_time:.4f} seconds")
    print(f"Sklearn K-means execution time: {sklearn_time:.4f} seconds")
    print(f"Custom K-means inertia: {custom_inertia:.2f}")
    print(f"Sklearn K-means inertia: {sklearn_kmeans.inertia_:.2f}")
    
    # Compare cluster assignments
    agreement = np.mean(np.array(custom_clusters) == sklearn_clusters)
    print(f"Cluster assignment agreement: {agreement:.2%}")
    
    # Analyze cluster sizes
    custom_sizes = [sum(np.array(custom_clusters) == i) for i in range(k)]
    sklearn_sizes = [sum(sklearn_clusters == i) for i in range(k)]
    print("Custom cluster sizes:", custom_sizes)
    print("Sklearn cluster sizes:", sklearn_sizes)
    
    return custom_clusters, custom_centroids, sklearn_clusters, sklearn_kmeans.cluster_centers_

def elbow_method(points: np.ndarray, max_clusters: int = 10, max_iters: int = 100) -> List[float]:
    """
    Perform the elbow method to find the optimal number of clusters.

    This function calculates the inertia for different numbers of clusters
    and returns a list of inertias to help determine the optimal k.

    Args:
        points (np.ndarray): Input data points.
        max_clusters (int, optional): Maximum number of clusters to test. Defaults to 10.
        max_iters (int, optional): Maximum number of iterations for each k-means run. Defaults to 100.

    Returns:
        List[float]: List of inertias for each k value.
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        clusters, centroids = kmeans(points, k, max_iters=max_iters)
        inertia = sum(min(euclidean_distance(point, centroid)**2 for centroid in centroids) for point in points)
        inertias.append(inertia)
        print(f"k={k}: inertia={inertia:.2f}")
    return inertias