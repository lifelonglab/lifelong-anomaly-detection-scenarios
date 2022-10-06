import numpy as np

from data.utils import load_npy_dataset
from prepare_scenario import ScenarioConfig, prepare_scenario

if __name__ == '__main__':
    dataset_name = 'unsw'
    normal_data, anomaly_data = load_npy_dataset('data/unsw/full_unsw.npy')

    config = ScenarioConfig(scenario_type='random_anomalies', clusters_no=3, size_per_cluster=5000)

    concepts = prepare_scenario(normal_data, anomaly_data, config)

    np.save(
        f'out/{dataset_name}_{config.scenario_type}_concepts_{config.clusters_no}_per_cluster_{config.size_per_cluster}',
        concepts)
