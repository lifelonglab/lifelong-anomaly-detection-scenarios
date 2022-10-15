import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

columns_to_encode = [1, 2, 3]

training = pd.read_csv('original/KDDTrain+.txt', header=None)
testing = pd.read_csv('original/KDDTest+.txt', header=None)

all_data = pd.concat([training, testing], ignore_index=True)
all_data = all_data.rename(columns={41: 'Label', 42: 'Difficulty'}, errors='raise')

# encode string columns
for column in columns_to_encode:
    c_data = all_data[column]
    encoder = OrdinalEncoder()
    all_data[column] = encoder.fit_transform(c_data.values.reshape((len(all_data), 1))).reshape((len(all_data)))

binary_labels = np.array([0 if label == 'normal' else 1 for label in all_data['Label'].values])
del all_data['Label']
del all_data['Difficulty']

all_data = MinMaxScaler().fit_transform(all_data.values)
print(all_data.shape)
all_data_with_labels = np.append(all_data, binary_labels.reshape((len(binary_labels), 1)), axis=1)
print(all_data_with_labels.shape)

np.save('full_nsl', all_data_with_labels)