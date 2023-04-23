import numpy as np

path = 'out/wind/wind_random_anomalies_3_concepts_25000_per_cluster.npy'

loaded_data = np.load(path, allow_pickle=True)

for concept in loaded_data:
    print(f'Concept: {concept.name} | train size: {concept.train_data.shape} | test size {concept.test_data.shape} '
          f'and test labels {concept.test_labels.shape}')
