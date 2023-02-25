import pathlib
from typing import get_args

import numpy as np

from data.utils import load_npy_dataset
from prepare_scenario import prepare_scenario
from scenario_config import ScenarioConfig, ScenarioType

if __name__ == '__main__':
    # dataset_name = 'unsw'
    # normal_data, anomaly_data = load_npy_dataset('data/unsw/full_unsw.npy')
    #
    # dataset_name = 'nsl-kdd'
    # normal_data, anomaly_data = load_npy_dataset('data/nsl_kdd/full_nsl.npy')

    dataset_name = 'wind'
    normal_data, anomaly_data = load_npy_dataset('data/wind/full_wind.npy')

    # dataset_name = 'energy'
    # normal_data, anomaly_data = load_npy_dataset('data/energy/full_energy.npy')

    for scenario_type in get_args(ScenarioType):
        print(scenario_type)
        config = ScenarioConfig(scenario_type=scenario_type, clusters_no=3, size_per_cluster=25_000)

        concepts = prepare_scenario(normal_data, anomaly_data, config)
        pathlib.Path(f'out/{dataset_name}').mkdir(parents=True, exist_ok=True)
        np.save(
            f'out/{dataset_name}/{dataset_name}_{config.scenario_type}_{config.clusters_no}_concepts_{config.size_per_cluster}_per_cluster',
            concepts)
