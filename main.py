import numpy as np

from data.utils import load_npy_dataset
from prepare_scenario import prepare_scenario
from scenario_config import ScenarioConfig

if __name__ == '__main__':
    dataset_name = 'unsw'
    normal_data, anomaly_data = load_npy_dataset('data/unsw/full_unsw.npy')

    config = ScenarioConfig(scenario_type='random_anomalies', clusters_no=10, size_per_cluster=5000)

    concepts = prepare_scenario(normal_data, anomaly_data, config)

    np.save(
        f'out/{dataset_name}_{config.scenario_type}_{config.clusters_no}_concepts_{config.size_per_cluster}_per_cluster',
        concepts)
