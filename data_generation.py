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
