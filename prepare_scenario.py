import pathlib
from typing import List

import numpy as np
from loguru import logger

from clustering import (
    create_anomaly_clusters_closest_to_normal,
    create_anomaly_clusters_randomly_assigned,
    create_concepts,
    create_random_anomaly_clusters,
)
from concept import Concept
from csv_writer import save_scenario_as_csv
from scenario_config import ScenarioConfig


def split_into_train_test(normal_cluster, anomaly_cluster, normal_ids, anomaly_ids, desired_ratio=0.5):
    split_point = int(3 * len(normal_cluster) / 5)
    train_data_normal = np.array(normal_cluster[:split_point])
    train_ids = normal_ids[:split_point]

    test_data_normal = np.array(normal_cluster[split_point:])
    test_ids_normal = normal_ids[split_point:]
    n_normal_test = len(test_data_normal)

    n_anomaly = int(n_normal_test * desired_ratio / (1 - desired_ratio))
    n_anomaly = min(n_anomaly, len(anomaly_cluster))
    test_data_anomaly = np.array(anomaly_cluster[:n_anomaly])
    test_ids_anomaly = anomaly_ids[:n_anomaly]

    test_data_normal_with_labels = np.append(test_data_normal, np.zeros((n_normal_test, 1)), axis=1)
    test_data_anomaly_with_labels = np.append(test_data_anomaly, np.ones((len(test_data_anomaly), 1)), axis=1)

    test_data_with_labels = np.concatenate((test_data_normal_with_labels, test_data_anomaly_with_labels))
    test_ids = np.concatenate((test_ids_normal, test_ids_anomaly))

    # shuffle test data and test ids synchronously
    indices = np.arange(len(test_data_with_labels))
    np.random.shuffle(indices)
    test_data_with_labels = test_data_with_labels[indices]
    test_ids = test_ids[indices]

    test_data, test_labels = test_data_with_labels[:, :-1], test_data_with_labels[:, -1]

    return train_data_normal, test_data, test_labels, train_ids, test_ids


def _create_anomaly_clusters(normal_clusters, anomaly_data, config: ScenarioConfig, sample_ids):
    anomalies_no_per_cluster = min(int(len(anomaly_data) / len(normal_clusters)), int(2 * config.size_per_concept / 5))

    if config.scenario_type == 'random_anomalies':
        return create_random_anomaly_clusters(
            anomaly_data, clusters_no=config.concepts_no,size_per_cluster=anomalies_no_per_cluster, sample_ids=sample_ids)
    elif config.scenario_type == 'clustered_with_random_assignment':
        return create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster, sample_ids=sample_ids)
    elif config.scenario_type == 'clustered_with_closest_assignment':
        return create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster, sample_ids=sample_ids)


def prepare_scenario(normal_data: np.ndarray, anomaly_data: np.ndarray, config: ScenarioConfig) -> List[Concept]:
    normal_clusters, normal_ids = create_concepts(
        data=normal_data[:, 1:],  # 1st column represents samples' identifiers
        concepts_no=config.concepts_no,
        size_per_concept=config.size_per_concept,
        sample_ids=normal_data[:, 0].astype(int)
    )

    anomaly_clusters, anomaly_ids, = _create_anomaly_clusters(
        normal_clusters=normal_clusters,
        anomaly_data=anomaly_data[:, 1:],  # 1st column represents samples' identifiers
        config=config,
        sample_ids=anomaly_data[:, 0].astype(int)
    )

    concepts = []
    for i, (normal_cluster, normal_ids_cluster, anomaly_cluster, anomaly_ids_cluster) in enumerate(
            zip(normal_clusters, normal_ids, anomaly_clusters, anomaly_ids)
    ):
        train_data, test_data, test_labels, train_ids, test_ids = split_into_train_test(
            normal_cluster, anomaly_cluster, normal_ids_cluster, anomaly_ids_cluster
        )
        logger.info(f"Extracting Concept_{i}: {len(train_data)} train and {len(test_data)} test samples")
        concepts.append(
            Concept(
                name=f'Concept_{i}',
                train_data=train_data,
                test_data=test_data,
                test_labels=test_labels,
                train_ids=train_ids,
                test_ids=test_ids
            )
        )

    return concepts

def prepare_and_save_scenario(
        dataset_name: str, normal_data: np.ndarray, anomaly_data: np.ndarray, config: ScenarioConfig
) -> None:
    concepts = prepare_scenario(normal_data, anomaly_data, config)
    path = pathlib.Path(
        f'out/{dataset_name}/{config.scenario_type}/{config.concepts_no}_concepts_'
        f'{config.size_per_concept}_per_cluster'
    )
    path.mkdir(exist_ok=True, parents=True)

    save_scenario_as_csv(scenario=concepts, config=config, output_dir=path)
