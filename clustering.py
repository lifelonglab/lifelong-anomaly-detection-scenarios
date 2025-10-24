import random
from typing import List, Tuple

import numpy as np
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


def create_concepts(
        data: np.ndarray, concepts_no: int, size_per_concept: int, sample_ids: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    clusters_id, distances = cluster_kmeans(data, concepts_no)

    concepts, identifiers = [], []

    for concept_id in range(concepts_no):
        cluster_data_with_distances = [
            (data, sample_id, dists[cluster_id])
            for cluster_id, data, dists, sample_id in zip(clusters_id, data, distances, sample_ids)
            if cluster_id == concept_id
        ]

        distances_for_cluster = [distance for _, _, distance in cluster_data_with_distances]
        sorted_indices = np.argsort(distances_for_cluster)

        sorted_data = np.array([cluster_data_with_distances[i][0] for i in sorted_indices])
        sorted_ids = np.array([cluster_data_with_distances[i][1] for i in sorted_indices])

        n_select = min(size_per_concept, len(sorted_data))
        selected_data = sorted_data[:n_select]
        selected_ids = sorted_ids[:n_select]

        indices = np.arange(n_select)
        np.random.shuffle(indices)
        selected_data = selected_data[indices]
        selected_ids = selected_ids[indices]

        concepts.append(selected_data)
        identifiers.append(selected_ids)

    return concepts, identifiers


def _find_closest_cluster(base_cluster_centroid: np.ndarray, clusters: List[Tuple[np.ndarray, np.ndarray]]):
    distances = [(i, np.linalg.norm(base_cluster_centroid - c_centroid)) for i, (c_data, c_centroid) in
                 enumerate(clusters)]
    # distances = [(i, wassertein_distance(np.array(base_cluster.data), np.array(c.data))) for i, c in enumerate(clusters)]
    distances.sort(key=lambda x: x[1])
    c_id = distances[0][0]
    return c_id, clusters[c_id][0]


def _calculate_centroid(c):
    return np.mean(c, axis=0)


def _reassign_clusters(anomaly_clusters, normal_clusters, sample_ids):
    normal_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in normal_clusters]
    for j, (_, c) in enumerate(normal_clusters_with_centroids):
        for k, (_, c1) in enumerate(normal_clusters_with_centroids):
            print(j, k, np.linalg.norm(c - c1))

    anomaly_clusters_with_ids = list(zip(anomaly_clusters, sample_ids))
    anomaly_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in anomaly_clusters]

    left_anomaly_clusters = anomaly_clusters_with_centroids
    left_anomaly_clusters_ids = anomaly_clusters_with_ids.copy()

    sorted_anomaly_ids = []
    sorted_anomaly_clusters = []

    for i, (c_data, centroid) in enumerate(normal_clusters_with_centroids):
        anomaly_cluster_id, anomalies = _find_closest_cluster(centroid, left_anomaly_clusters)
        sorted_anomaly_clusters.append(anomalies)
        sorted_anomaly_ids.append(left_anomaly_clusters_ids[anomaly_cluster_id][1])
        left_anomaly_clusters = [c for i, c in enumerate(left_anomaly_clusters) if i != anomaly_cluster_id]
        left_anomaly_clusters_ids = [c for j, c in enumerate(left_anomaly_clusters_ids) if j != anomaly_cluster_id]

    return sorted_anomaly_clusters, sorted_anomaly_ids


def create_random_anomaly_clusters(anomaly_data, clusters_no, size_per_cluster, sample_ids):
    indices = np.arange(len(anomaly_data))
    np.random.shuffle(indices)

    anomaly_data = anomaly_data[indices]
    sample_ids = np.array(sample_ids)[indices]

    clusters_data = [
        anomaly_data[size_per_cluster * i: size_per_cluster * (i + 1)]
        for i in range(clusters_no)
    ]

    clusters_ids = [
        sample_ids[size_per_cluster * i: size_per_cluster * (i + 1)]
        for i in range(clusters_no)
    ]

    return clusters_data, clusters_ids


def create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no, size_per_cluster, sample_ids):
    clusters_data, clusters_ids = create_concepts(
        anomaly_data,
        concepts_no=clusters_no,
        size_per_concept=size_per_cluster,
        sample_ids=sample_ids
    )

    combined = list(zip(clusters_data, clusters_ids))
    random.shuffle(combined)
    clusters_data, clusters_ids = zip(*combined)
    return list(clusters_data), list(clusters_ids)


def create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no, size_per_cluster, sample_ids):
    clusters_data, clusters_ids = create_concepts(
        anomaly_data,
        concepts_no=clusters_no,
        size_per_concept=size_per_cluster,
        sample_ids=sample_ids
    )

    reassigned_data, reassigned_ids = _reassign_clusters(
        anomaly_clusters=clusters_data,
        normal_clusters=normal_clusters,
        sample_ids=clusters_ids,
    )

    return reassigned_data, reassigned_ids
