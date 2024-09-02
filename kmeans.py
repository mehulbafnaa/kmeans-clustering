import numpy as np
from typing import List, Tuple, Optional

Point = np.ndarray
Centroid = np.ndarray
Cluster = List[int]

def euclidean_distance(p1: Point, p2: Point) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

def assign_to_clusters(points: Point, centroids: List[Centroid]) -> List[Cluster]:
    """
    Assign each point to the nearest centroid.

    Args:
        points (Point): Array of data points.
        centroids (List[Centroid]): List of current centroid positions.

    Returns:
        List[Cluster]: List of cluster assignments for each point.
    """
    distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in points])
    return [np.where(row == np.min(row))[0][0] for row in distances]

def update_centroids(points: np.ndarray, clusters: List[int], k: int) -> List[Centroid]:
    """
    Update centroid positions based on the mean of assigned points.

    Args:
        points (np.ndarray): Array of data points.
        clusters (List[int]): Current cluster assignments.
        k (int): Number of clusters.

    Returns:
        List[Centroid]: Updated centroid positions.
    """
    centroids = []
    for i in range(k):
        cluster_points = points[np.array(clusters) == i]
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))
        else:
            centroids.append(points[np.random.choice(len(points))])
    return centroids

def has_converged(old_centroids: List[Centroid], new_centroids: List[Centroid], tol: float) -> bool:
    """
    Check if the algorithm has converged based on centroid movement.

    Args:
        old_centroids (List[Centroid]): Previous centroid positions.
        new_centroids (List[Centroid]): Updated centroid positions.
        tol (float): Convergence tolerance.

    Returns:
        bool: True if converged, False otherwise.
    """
    return all(euclidean_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))

def kmeans_plus_plus_init(points: np.ndarray, k: int) -> List[Centroid]:
    """
    Initialize centroids using the k-means++ method.

    This method chooses initial centroids that are far apart from each other,
    which can lead to better and more consistent clustering results.

    Args:
        points (np.ndarray): Array of data points.
        k (int): Number of clusters.

    Returns:
        List[Centroid]: Initial centroid positions.
    """
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
    """
    Perform k-means clustering on the input points.

    This function implements the k-means clustering algorithm, which aims to partition
    n observations into k clusters in which each observation belongs to the cluster
    with the nearest mean (centroid).

    Args:
        points (np.ndarray): Input data points to be clustered.
        k (int): Number of clusters to form.
        max_iters (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Convergence tolerance. Defaults to 1e-5.
        init_method (str, optional): Initialization method ('kmeans++' or 'random'). Defaults to 'kmeans++'.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[List[int], List[Centroid]]: A tuple containing:
            - List of cluster assignments for each point.
            - List of final centroid positions.

    Raises:
        ValueError: If k is greater than the number of data points or if an invalid initialization method is specified.
    """
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