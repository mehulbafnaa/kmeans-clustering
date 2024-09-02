
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