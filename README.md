import random
from typing import List, Sequence, Tuple

def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two same-length points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def calculate_centroid(points: List[Sequence[float]]) -> List[float]:
    """Centroid of a non-empty cluster of points."""
    if not points:
        raise ValueError("Cannot compute centroid of an empty cluster.")
    n_dims = len(points[0])
    sums = [0.0] * n_dims
    for pt in points:
        for i in range(n_dims):
            sums[i] += pt[i]
    return [s / len(points) for s in sums]

def assign_to_clusters(data: List[Sequence[float]], centroids: List[Sequence[float]]):
    """Assign each point to its nearest centroid."""
    clusters = {i: [] for i in range(len(centroids))}
    for point in data:
        nearest = min(range(len(centroids)),
                      key=lambda i: euclidean_distance(point, centroids[i]))
        clusters[nearest].append(point)
    return clusters

def update_centroids(clusters, data):
    """Recalculate centroids; reinit empty clusters to a random data point."""
    new_centroids = []
    for i in sorted(clusters.keys()):
        if clusters[i]:
            new_centroids.append(calculate_centroid(clusters[i]))
        else:
            new_centroids.append(random.choice(data))
    return new_centroids

def kmeans(data: List[Sequence[float]], k: int, max_iterations: int = 100, tol: float = 1e-6):
    """
    Performs K-means clustering.
    Returns (final_centroids, final_clusters).
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(data):
        raise ValueError("k cannot be greater than the number of data points.")

    centroids = random.sample(data, k)

    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters, data)

        if all(euclidean_distance(c, n) < tol for c, n in zip(centroids, new_centroids)):
            return new_centroids, clusters
        centroids = new_centroids

    return centroids, clusters

if __name__ == "__main__":
    # Small demo
    data = [
        (1.0, 2.0), (1.2, 1.9), (0.8, 2.1),
        (5.0, 8.0), (5.3, 7.9), (4.8, 8.2),
        (9.0, 1.0), (9.3, 1.2), (8.7, 0.8),
    ]
    k = 3
    centers, clusters = kmeans(data, k)
    print("Centroids:", centers)
    for i, pts in clusters.items():
        print(f"Cluster {i}: {pts}")
