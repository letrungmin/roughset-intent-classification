import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class RoughSetClassifier:
    def __init__(self, n_clusters=150, lower_threshold=0.85):
        """
        n_clusters: Number of knowledge granules (clusters). Larger numbers create finer granules.
        lower_threshold: Purity threshold. If > 85% of vectors in a cluster belong to one label, 
                         the cluster is classified as Certain (Lower Approximation).
        """
        self.n_clusters = n_clusters
        self.lower_threshold = lower_threshold
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        
        # Stores Rough Set structure for each cluster
        # Format: {cluster_id: {'type': 'CERTAIN'/'BOUNDARY', 'labels': [...]}}
        self.granules_info = {}

    def fit(self, X, y):
        """
        Build the Rough Set space from training data.
        X: Feature matrix (e.g., 1000x384)
        y: Corresponding labels (e.g., Intent IDs)
        """
        print(f"Generating {self.n_clusters} knowledge granules using K-Means...")
        cluster_labels = self.kmeans.fit_predict(X)
        
        y = np.array(y)
        
        # Analyze the purity of each granule
        for cluster_id in range(self.n_clusters):
            # Get all actual labels for samples falling into this cluster
            idx_in_cluster = np.where(cluster_labels == cluster_id)[0]
            labels_in_cluster = y[idx_in_cluster]
            
            if len(labels_in_cluster) == 0:
                continue
                
            # Calculate label distribution within the cluster
            label_counts = Counter(labels_in_cluster)
            total_samples = len(labels_in_cluster)
            most_common_label, most_common_count = label_counts.most_common(1)[0]
            
            purity = most_common_count / total_samples
            
            if purity >= self.lower_threshold:
                # Lower Approximation (Certain Region)
                self.granules_info[cluster_id] = {
                    'type': 'CERTAIN',
                    'labels': [most_common_label],
                    'purity': purity
                }
            else:
                # Boundary Region (Ambiguous) - Include labels with > 10% representation
                boundary_labels = [lbl for lbl, count in label_counts.items() if (count/total_samples) > 0.1]
                self.granules_info[cluster_id] = {
                    'type': 'BOUNDARY',
                    'labels': boundary_labels,
                    'purity': purity
                }
        print("Successfully constructed Lower Approximations and Boundary Regions.")

    def predict(self, X):
        """
        Predict labels for new query sentences.
        Returns: List of tuples (Predicted Labels, Region Type)
        """
        cluster_preds = self.kmeans.predict(X)
        results = []
        for cid in cluster_preds:
            info = self.granules_info.get(cid, {'type': 'UNKNOWN', 'labels': [-1]})
            results.append((info['labels'], info['type']))
        return results