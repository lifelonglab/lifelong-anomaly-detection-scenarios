import random
from dataclasses import dataclass

import numpy as np
from numpy.random import shuffle
from sklearn.cluster import KMeans

def _create_kmeans(n_clusters):
    return KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                  random_state=42, copy_x=True, algorithm='auto')


def cluster_kmeans(data, clusters_no):
    """ Data should not contain labels """
    kmeans_model = _create_kmeans(clusters_no)
    clustering_results = kmeans_model.fit_predict(data)
    distances = kmeans_model.transform(data)
    return clustering_results, distances


def create_clusters(data, clusters_no, size_per_cluster):
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


def create_random_anomaly_clusters(anomaly_data, clusters_no, size_per_cluster):
    random.shuffle(anomaly_data)
    return [anomaly_data[size_per_cluster * i: size_per_cluster * (i+1)] for i in range(clusters_no)]


def create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no, size_per_cluster):
    anomalies_clusters = create_clusters(anomaly_data, clusters_no=clusters_no, size_per_cluster=size_per_cluster)
