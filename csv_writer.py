import pathlib
from typing import List

import pandas as pd
from loguru import logger

from concept import Concept
from scenario_config import ScenarioConfig


def save_scenario_as_csv(
    scenario: List[Concept], config: ScenarioConfig, output_dir: pathlib.Path
) -> None:
    train_dfs = []
    test_dfs = []

    for i, concept in enumerate(scenario):
        train_df = pd.DataFrame(
            concept.train_data,
            columns=[f"feature_{i}" for i in range(concept.train_data.shape[1])],
        )
        train_df["label"] = 0
        train_df["concept_name"] = concept.name
        train_df["concept_id"] = i
        train_df["sample_id"] = concept.train_ids

        test_df = pd.DataFrame(
            concept.test_data,
            columns=[f"feature_{i}" for i in range(concept.test_data.shape[1])],
        )
        test_df["label"] = concept.test_labels
        test_df["concept_name"] = concept.name
        test_df["concept_id"] = i
        test_df["sample_id"] = concept.test_ids

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Output stored in {output_dir}")
