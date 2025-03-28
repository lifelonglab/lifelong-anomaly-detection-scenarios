import pathlib
from typing import List

import pandas as pd

from concept import Concept


def save_scenario_as_csv(scenario: List[Concept], output_path: pathlib.Path):
    dfs = []
    for i, concept in enumerate(scenario):
        train_df = pd.DataFrame(concept.train_data, columns=[f'feature_{i}' for i in range(concept.train_data.shape[1])])
        train_df['label'] = 0
        train_df['concept_name'] = concept.name
        train_df['concept_id'] = i

        test_df = pd.DataFrame(concept.test_data, columns=[f'feature_{i}' for i in range(concept.test_data.shape[1])])
        test_df['label'] = concept.test_labels
        test_df['concept_name'] = concept.name
        test_df['concept_id'] = i

        dfs.append(train_df)
        dfs.append(test_df)

    final_df = pd.concat(dfs)
    final_df.to_csv(output_path, index=False)
