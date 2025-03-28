import pathlib
from typing import List

import numpy as np
from numpy.random import shuffle

from clustering import create_concepts, create_random_anomaly_clusters, create_anomaly_clusters_randomly_assigned, \
    create_anomaly_clusters_closest_to_normal
from concept import Concept
from csv_writer import save_scenario_as_csv
from scenario_config import ScenarioConfig


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


def _create_anomaly_clusters(normal_clusters, anomaly_data, config: ScenarioConfig):
    anomalies_no_per_cluster = min(int(len(anomaly_data) / len(normal_clusters)), int(2 * config.size_per_concept / 5))

    if config.scenario_type == 'random_anomalies':
        return create_random_anomaly_clusters(anomaly_data, clusters_no=config.concepts_no,
                                              size_per_cluster=anomalies_no_per_cluster)
    elif config.scenario_type == 'clustered_with_random_assignment':
        return create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster)
    elif config.scenario_type == 'clustered_with_closest_assignment':
        return create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster)


def prepare_scenario(normal_data: np.ndarray, anomaly_data: np.ndarray, config: ScenarioConfig) -> List[Concept]:
    normal_clusters = create_concepts(normal_data, concepts_no=config.concepts_no,
                                      size_per_concept=config.size_per_concept)

    anomalies_clusters = _create_anomaly_clusters(normal_clusters, anomaly_data, config=config)

    concepts = []
    for i, (normal_cluster, anomaly_cluster) in enumerate(zip(normal_clusters, anomalies_clusters)):
        train_data, test_data, test_labels = split_into_train_test(normal_cluster, anomaly_cluster)
        concepts.append(
            Concept(name=f'Concept_{i}', train_data=train_data, test_data=test_data, test_labels=test_labels))

    return concepts


def prepare_and_save_scenario(scenario_name: str, normal_data: np.ndarray, anomaly_data: np.ndarray,
                              config: ScenarioConfig):
    concepts = prepare_scenario(normal_data, anomaly_data, config)
    path = pathlib.Path(f'out/{scenario_name}/')
    path.mkdir(exist_ok=True, parents=True)
    output_filename = f'{scenario_name}_{config.scenario_type}_{config.concepts_no}_concepts_{config.size_per_concept}_per_cluster'
    np.save(str(path / output_filename), np.array(concepts))

    save_scenario_as_csv(scenario=concepts, output_path=path / f'{output_filename}.csv')
