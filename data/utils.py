from typing import Tuple

import numpy as np


def load_npy_dataset(filepath) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(filepath, allow_pickle=True)
    normal_data = data[data[:, -1] == 0][:, :-1]
    anomaly_data = data[data[:, -1] == 1][:, :-1]
    return normal_data, anomaly_data
