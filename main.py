import pathlib
from typing import get_args

import numpy as np

from data.utils import load_npy_dataset
from prepare_scenario import prepare_scenario, prepare_and_save_scenario
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
        config = ScenarioConfig(scenario_type=scenario_type, concepts_no=3, size_per_concept=25_000)

        prepare_and_save_scenario(dataset_name, normal_data, anomaly_data, config)
