import numpy as np
import h5py
import math
import torch
from torch.distributions import Categorical

from utils.transforms import to_one_hot_majority, to_one_hot, concatenate_cities_labels, concatenate_cities_patches

# Define labels
labels = np.arange(1,18)

path_data = "/data/lcz42_votes/data/"

test_cities = ['amsterdam', 'berlin', 'cologne', 'london', 'madrid',
             'milan', 'munich', 'paris', 'rome', 'zurich']

test_addons = ['guangzhou_addon', 'islamabad_addon', 'jakarta_addon', 'losangeles_addon', 'moscow_addon',
              'mumbai_addon', 'nairobi_addon', 'riodejaneiro_addon']

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities_majority = to_one_hot_majority(y_test_cities, labels, addon=False)
y_test_cities = to_one_hot(y_test_cities, labels, addon=False)
# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons_majority = to_one_hot_majority(y_test_addons, labels, addon=True)
y_test_addons = to_one_hot(y_test_addons, labels, addon=True)

# Concatenate labels of test cities & addons
y_test = np.vstack((y_test_cities, y_test_addons))
y_test = y_test.astype(int)

# Concatenate labels of test cities & addons after majority vote
y_test_majority = np.vstack((y_test_cities_majority, y_test_addons_majority))
y_test_majority = y_test_majority.astype(int)

np.random.seed(4242)
indices_val = np.random.choice(np.arange(y_test.shape[0]), math.ceil(0.3 * y_test.shape[0]), False)
indices_test = list(set(np.arange(y_test.shape[0])) - set(indices_val))

y_test = y_test[indices_test,]
y_test_majority = y_test_majority[indices_test, ]

indices_test = np.where(np.where(y_test_majority == np.amax(y_test_majority, 0))[1] + 1 < 11)[0]
# !!! Limit to 10 columns to receive distribution + entropy only w.r.t. urban classes !!!
y_test = y_test[indices_test, :10]

y_test_label_distributions = y_test / y_test.sum(axis=1, keepdims=True)

path_data = "/data/lcz42_cities/"

test_label_distributions_data_h5 = h5py.File(path_data + 'test_label_distributions_data.h5', 'w')
test_label_distributions_data_h5.create_dataset('test_label_distributions', data=y_test_label_distributions)
test_label_distributions_data_h5.close()

y_test = torch.from_numpy(y_test)

entropies_test = Categorical(probs = y_test).entropy()

entropies_h5 = h5py.File(path_data + 'entropies_test.h5', 'w')
entropies_h5.create_dataset('entropies_test', data=entropies_test.numpy())
entropies_h5.close()