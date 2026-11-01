"""
Module 6: Unsupervised Learning & Feature Clustering
=====================================================
Focus: K-Means clustering to partition vehicles into Economy, Mid-Range,
and Luxury segments as an auxiliary feature.
Includes:
- K-Means algorithm implementation (Lloyd's heuristic)
- Cluster centroid estimation
- Segment labeling ("Economy Commuter", "Mid-Range Executive", "Luxury Premium")
- Auxiliary cluster distance feature generation
"""

import math
import random
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module6")


class KMeansCustom:
    """
    K-Means Clustering for automotive market segmentation.
    Minimizes within-cluster sum of squares (WCSS / Inertia):
    J = sum_k sum_{x in S_k} ||x - mu_k||^2
    """
    SEGMENT_NAMES = {
        0: "Economy Commuter",
        1: "Mid-Range Executive",
        2: "Luxury Premium"
    }

    def __init__(self, k: int = 3, max_iter: int = 50):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.cluster_labels_map = {}

    @staticmethod
    def euclidean_dist(p1: List[float], p2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def fit(self, X: List[List[float]], unscaled_prices: List[float] = None):
        n_samples = len(X)
        if n_samples < self.k:
            return self

        # Initialize centroids randomly from data samples
        random.seed(42)
        initial_indices = random.sample(range(n_samples), self.k)
        self.centroids = [list(X[i]) for i in initial_indices]

        labels = [0] * n_samples

        for _ in range(self.max_iter):
            # Step 1: Assign points to nearest centroid
            new_labels = []
            for point in X:
                distances = [self.euclidean_dist(point, c) for c in self.centroids]
                nearest = distances.index(min(distances))
                new_labels.append(nearest)

            # Step 2: Check convergence
            if new_labels == labels:
                break
            labels = new_labels

            # Step 3: Recompute centroids
            for cluster_idx in range(self.k):
                cluster_points = [X[i] for i in range(n_samples) if labels[i] == cluster_idx]
                if cluster_points:
                    dim = len(X[0])
                    self.centroids[cluster_idx] = [
                        sum(p[d] for p in cluster_points) / len(cluster_points) for d in range(dim)
                    ]

        # Order clusters by average price to map to Economy, Mid-Range, Luxury
        if unscaled_prices:
            cluster_avg_prices = []
            for c_idx in range(self.k):
                c_prices = [unscaled_prices[i] for i in range(n_samples) if labels[i] == c_idx]
                avg_p = sum(c_prices) / len(c_prices) if c_prices else 0.0
                cluster_avg_prices.append((c_idx, avg_p))

            # Sort ascending: lowest price is Economy (0), middle is Mid-Range (1), highest is Luxury (2)
            cluster_avg_prices.sort(key=lambda x: x[1])
            self.cluster_labels_map = {
                cluster_avg_prices[0][0]: "Economy Commuter",
                cluster_avg_prices[1][0]: "Mid-Range Executive",
                cluster_avg_prices[2][0]: "Luxury Premium"
            }
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        """Assigns samples to closest cluster index."""
        preds = []
        for point in X:
            dists = [self.euclidean_dist(point, c) for c in self.centroids]
            preds.append(dists.index(min(dists)))
        return preds

    def get_segment_name(self, point: List[float]) -> str:
        """Returns the market segment name for an input feature vector."""
        if not self.centroids:
            return "Mid-Range Executive"
        dists = [self.euclidean_dist(point, c) for c in self.centroids]
        closest_idx = dists.index(min(dists))
        return self.cluster_labels_map.get(closest_idx, self.SEGMENT_NAMES.get(closest_idx, "Standard"))
