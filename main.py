from typing import get_args

import numpy as np
from loguru import logger

from prepare_scenario import prepare_and_save_scenario
from scenario_config import ScenarioConfig, ScenarioType

if __name__ == "__main__":
    dataset_name = "insdn"
    normal_data = np.load(f"data/{dataset_name}/insdn_normal.npy")
    anomaly_data = np.load(f"data/{dataset_name}/insdn_anomaly.npy")

    # === Configuration flag ===
    # Set to True if the loaded .npy files already contain a first column representing unique sample identifiers.
    # This identifier is required to track which specific samples are assigned to each concept.
    # If False, unique IDs will be automatically generated and appended as the first column.
    has_sample_ids = False  # ← change this flag depending on your dataset

    if not has_sample_ids:
        normal_data = np.hstack(
            [np.arange(len(normal_data)).reshape(-1, 1), normal_data]
        )
        anomaly_data = np.hstack(
            [
                np.arange(
                    len(normal_data), len(normal_data) + len(anomaly_data)
                ).reshape(-1, 1),
                anomaly_data,
            ]
        )

    for scenario_type in get_args(ScenarioType):
        logger.info(f"Running scenario {scenario_type} for {dataset_name} dataset...")
        config = ScenarioConfig(
            scenario_type=scenario_type, concepts_no=3, size_per_concept=22_000
        )
        prepare_and_save_scenario(dataset_name, normal_data, anomaly_data, config)
        logger.success(f"Finished scenario {scenario_type}!")
