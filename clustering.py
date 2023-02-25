import random
from typing import List, Tuple

import numpy as np
from numpy.random import shuffle
from sklearn.cluster import KMeans


def _create_kmeans(n_clusters):
    return KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                  random_state=42, copy_x=True, algorithm='lloyd')


def cluster_kmeans(data: np.ndarray, clusters_no: int):
    """ Data should not contain labels """
    kmeans_model = _create_kmeans(clusters_no)
    clustering_results = kmeans_model.fit_predict(data)
    distances = kmeans_model.transform(data)
    return clustering_results, distances


def create_clusters(data: np.ndarray, clusters_no: int, size_per_cluster: int) -> List[np.ndarray]:
    clusters_id, distances = cluster_kmeans(data, clusters_no)

    concepts = []

    for c in range(clusters_no):
        # select data for cluster by sorting it by distance
        cluster_data_with_distances = [(d, dists[c_id]) for c_id, d, dists in
                                       zip(clusters_id, data, distances) if c_id == c]
        sorted_cluster_data = [d for d, _ in sorted(cluster_data_with_distances, key=lambda t: t[1])]
        selected_data = np.array(sorted_cluster_data[:min(size_per_cluster, len(sorted_cluster_data))])
        shuffle(selected_data)  # we don't want data to stay sorted
        concepts.append(selected_data)

    return concepts


def _find_closest_cluster(base_cluster_centroid: np.ndarray, clusters: List[Tuple[np.ndarray, np.ndarray]]):
    distances = [(i, np.linalg.norm(base_cluster_centroid - c_centroid)) for i, (c_data, c_centroid) in
                 enumerate(clusters)]
    # distances = [(i, wassertein_distance(np.array(base_cluster.data), np.array(c.data))) for i, c in enumerate(clusters)]
    distances.sort(key=lambda x: x[1])
    c_id = distances[0][0]
    return c_id, clusters[c_id][0]


def _calculate_centroid(c):
    return np.mean(c, axis=0)


def _reassign_clusters(anomaly_clusters, normal_clusters):
    normal_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in normal_clusters]
    for j, (_, c) in enumerate(normal_clusters_with_centroids):
        for k, (_, c1) in enumerate(normal_clusters_with_centroids):
            print(j, k, np.linalg.norm(c - c1))

    anomaly_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in anomaly_clusters]
    left_anomaly_clusters = anomaly_clusters_with_centroids
    sorted_anomaly_clusters = []
    for i, (c_data, centroid) in enumerate(normal_clusters_with_centroids):
        anomaly_cluster_id, anomalies = _find_closest_cluster(centroid, left_anomaly_clusters)
        sorted_anomaly_clusters.append(anomalies)
        left_anomaly_clusters = [c for i, c in enumerate(left_anomaly_clusters) if i != anomaly_cluster_id]
    return sorted_anomaly_clusters


def create_random_anomaly_clusters(anomaly_data, clusters_no, size_per_cluster):
    random.shuffle(anomaly_data)
    return [anomaly_data[size_per_cluster * i: size_per_cluster * (i + 1)] for i in range(clusters_no)]


def create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no, size_per_cluster):
    anomaly_clusters = create_clusters(anomaly_data, clusters_no=clusters_no, size_per_cluster=size_per_cluster)
    shuffle(anomaly_clusters)
    return anomaly_clusters


def create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no, size_per_cluster):
    anomaly_clusters = create_clusters(anomaly_data, clusters_no=clusters_no, size_per_cluster=size_per_cluster)
    return _reassign_clusters(anomaly_clusters=anomaly_clusters, normal_clusters=normal_clusters)
