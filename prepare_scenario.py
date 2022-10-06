from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.random import shuffle

from clustering import create_clusters, create_random_anomaly_clusters

ScenarioType = Literal['random_anomalies']


@dataclass
class ScenarioConfig:
    scenario_type: ScenarioType
    clusters_no: int
    size_per_cluster: int


def split_into_train_test(normal_cluster, anomaly_cluster):
    split_point = int(3 * len(normal_cluster.data) / 5)
    train_data_normal = np.array(normal_cluster.data[:split_point])
    test_data_normal = np.array(normal_cluster.data[split_point:])
    test_data_normal_with_labels = np.append(test_data_normal, np.zeros((len(test_data_normal), 1)), axis=1)
    test_data_anomaly_with_labels = np.append(anomaly_cluster, np.ones((len(anomaly_cluster), 1)), axis=1)

    test_data_with_labels = np.concatenate((test_data_normal_with_labels, test_data_anomaly_with_labels))
    shuffle(test_data_with_labels)

    test_data, test_labels = test_data_with_labels[:, :-1], test_data_with_labels[:, -1]

    return train_data_normal, test_data, test_labels


def prepare_scenario(normal_data, anomaly_data, config: ScenarioConfig):
    normal_clusters = create_clusters(normal_data, clusters_no=config.clusters_no,
                                      size_per_cluster=config.size_per_cluster)

    anomalies_no_per_cluster = min(int(len(anomaly_data) / len(normal_clusters)), int(2 * config.size_per_cluster / 5))
    anomalies_clusters = create_random_anomaly_clusters(anomaly_data, clusters_no=config.clusters_no,
                                                        size_per_cluster=anomalies_no_per_cluster)

    concepts = []
    for normal_cluster, anomaly_cluster in zip(normal_clusters, anomalies_clusters):
        train_data, test_data, test_labels = split_into_train_test(normal_cluster, anomaly_cluster)
        concepts.append({'train_data': train_data, 'test_data': test_data, 'test_labels': test_labels})

    return concepts
