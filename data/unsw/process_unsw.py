import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

columns_to_encode = ['proto', 'service', 'state']

training = pd.read_csv('original/UNSW_NB15_training-set.csv')
testing = pd.read_csv('original/UNSW_NB15_testing-set.csv')

# encode string columns
for column in columns_to_encode:
    encoder = OrdinalEncoder()
    training_data = training[column].values
    testing_data = testing[column].values
    all_data = np.concatenate([training_data, testing_data])
    encoder.fit(all_data.reshape((len(all_data), 1)))

    training[column] = encoder.transform(training_data.reshape((len(training[column]), 1)))
    testing[column] = encoder.transform(testing_data.reshape((len(testing[column]), 1)))

del training['label']
del training['attack_cat']
del testing['attack_cat']

# scale data using only training data to get min-max range
scaler = MinMaxScaler()
training = scaler.fit_transform(training.values)
testing_labels = testing.values[:, -1]
testing_values = scaler.transform(testing.values[:, :-1])

# merge all_data
training_labels = np.zeros((len(training), 1))
training_with_labels = np.append(training, training_labels, axis=1)

testing_with_labels = np.append(testing_values, testing_labels.reshape((len(testing_labels), 1)), axis=1)

all_data = np.concatenate([training_with_labels, testing_with_labels])

# save
np.save('full_unsw', all_data)
