from dataclasses import dataclass

import numpy as np


@dataclass
class Concept:
    name: str
    train_data: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
