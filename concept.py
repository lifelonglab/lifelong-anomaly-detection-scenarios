from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Concept:
    name: str
    train_data: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray
