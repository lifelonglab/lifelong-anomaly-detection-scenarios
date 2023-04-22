from typing import get_args

from data.utils import load_npy_dataset
from prepare_scenario import prepare_and_save_scenario
from scenario_config import ScenarioType, ScenarioConfig

DATASET_NAME = 'nsl-kdd'

CONFIGS_TO_GENERATE = [
    ScenarioConfig(scenario_type=scenario_type, concepts_no=clusters_no, size_per_concept=size_per_cluster)
    for scenario_type in get_args(ScenarioType)
    for clusters_no in [3, 10]
    for size_per_cluster in [5_000]
]


def _generate_nsl_kdd_scenario(config: ScenarioConfig) -> None:
    normal_data, anomaly_data = load_npy_dataset('data/nsl_kdd/full_nsl.npy')
    prepare_and_save_scenario(DATASET_NAME, normal_data, anomaly_data, config)


if __name__ == '__main__':
    for config in CONFIGS_TO_GENERATE:
        print(
            f"Generating scenario type {config.scenario_type} with {config.concepts_no} clusters and {config.size_per_concept} samples per normal cluster")
        _generate_nsl_kdd_scenario(config)
