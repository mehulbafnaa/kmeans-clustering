# File: kmeans.py

import numpy as np
from typing import List, Tuple, Optional

Point = np.ndarray
Centroid = np.ndarray
Cluster = List[int]

def euclidean_distance(p1: Point, p2: Point) -> float:
    return np.sqrt(np.sum((p1 - p2) ** 2))

def assign_to_clusters(points: Point, centroids: List[Centroid]) -> List[Cluster]:
    distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in points])
    return [np.where(row == np.min(row))[0][0] for row in distances]

def update_centroids(points: np.ndarray, clusters: List[int], k: int) -> List[Centroid]:
    centroids = []
    for i in range(k):
        cluster_points = points[np.array(clusters) == i]
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))
        else:
            centroids.append(points[np.random.choice(len(points))])
    return centroids

def has_converged(old_centroids: List[Centroid], new_centroids: List[Centroid], tol: float) -> bool:
    return all(euclidean_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))

def kmeans_plus_plus_init(points: np.ndarray, k: int) -> List[Centroid]:
    centroids = [points[np.random.choice(len(points))]]
    
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(point, centroid) for centroid in centroids]) for point in points])
        probabilities = distances ** 2 / np.sum(distances ** 2)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        i = np.searchsorted(cumulative_probabilities, r)
        centroids.append(points[i])
    
    return centroids

def kmeans(points: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-5, 
           init_method: str = 'kmeans++', random_state: Optional[int] = None) -> Tuple[List[int], List[Centroid]]:
    if random_state is not None:
        np.random.seed(random_state)
    
    if k > len(points):
        raise ValueError("k cannot be greater than the number of data points")
    
    if init_method == 'kmeans++':
        centroids = kmeans_plus_plus_init(points, k)
    elif init_method == 'random':
        centroids = points[np.random.choice(len(points), k, replace=False)]
    else:
        raise ValueError("Invalid initialization method. Choose 'kmeans++' or 'random'")
    
    for _ in range(max_iters):
        clusters = assign_to_clusters(points, centroids)
        new_centroids = update_centroids(points, clusters, k)
        
        if has_converged(centroids, new_centroids, tol):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# File: data_generation.py

import numpy as np

def create_custom_dataset(n_samples=1000, n_features=2, n_clusters=3, cluster_std=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    centers = np.random.randn(n_clusters, n_features) * 5  # Multiply by 5 to spread clusters
    
    X = np.vstack([
        np.random.randn(n_samples // n_clusters, n_features) * cluster_std + center
        for center in centers
    ])
    
    np.random.shuffle(X)  # Shuffle the dataset
    
    return X, centers

# File: visualization.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from kmeans import Centroid, Cluster

def plot_clusters(points: np.ndarray, clusters: List[Cluster], centroids: List[Centroid], title: str):
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
    plt.figure(figsize=(10, 7))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# File: analysis.py

import numpy as np
from typing import List
from kmeans import kmeans, euclidean_distance
from sklearn.cluster import KMeans as SklearnKMeans
import time

def compare_kmeans(points: np.ndarray, k: int, max_iters: int = 100):
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
    inertias = []
    for k in range(1, max_clusters + 1):
        clusters, centroids = kmeans(points, k, max_iters=max_iters)
        inertia = sum(min(euclidean_distance(point, centroid)**2 for centroid in centroids) for point in points)
        inertias.append(inertia)
        print(f"k={k}: inertia={inertia:.2f}")
    return inertias

# File: main.py

from data_generation import create_custom_dataset
from visualization import plot_clusters, plot_elbow
from analysis import compare_kmeans, elbow_method
import matplotlib.pyplot as plt

def main():
    # Generate custom sample data
    n_samples = 1000
    n_features = 2
    n_clusters = 4
    X, true_centers = create_custom_dataset(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters, cluster_std=1.0, random_state=42)
    
    # Plot the true centers
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.scatter(true_centers[:, 0], true_centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title("Custom Dataset with True Centers")
    plt.show()
    
    # Run Elbow method
    inertias = elbow_method(X, max_clusters=10)
    plot_elbow(range(1, 11), inertias)
    
    # Compare K-means implementations
    custom_clusters, custom_centroids, sklearn_clusters, sklearn_centroids = compare_kmeans(X, k=n_clusters, max_iters=100)
    
    # Plot results
    plot_clusters(X, custom_clusters, custom_centroids, "Custom K-means")
    plot_clusters(X, sklearn_clusters, sklearn_centroids, "Sklearn K-means")

if __name__ == "__main__":
    main()