

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