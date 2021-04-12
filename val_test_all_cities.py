import numpy as np
import h5py
import math

from utils.transforms import to_one_hot_majority, to_one_hot, concatenate_cities_labels, concatenate_cities_patches

# Define labels
labels = np.arange(1,18)

path_data = "/data/lcz42_votes/data/"
#path_data = "D:/Data/LCZ_Votes/"

test_cities = ['amsterdam', 'berlin', 'cologne', 'london', 'madrid',
             'milan', 'munich', 'paris', 'rome', 'zurich']

test_addons = ['guangzhou_addon', 'islamabad_addon', 'jakarta_addon', 'losangeles_addon', 'moscow_addon',
              'mumbai_addon', 'nairobi_addon', 'riodejaneiro_addon']

x_test = concatenate_cities_patches(test_cities + test_addons)

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities = to_one_hot_majority(y_test_cities, labels, addon=False)
# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons = to_one_hot_majority(y_test_addons, labels, addon=True)

# Concatenate labels of test cities & addons
y_test = np.vstack((y_test_cities, y_test_addons))
y_test = y_test.astype(int)

np.random.seed(4242)
indices_val = np.random.choice(np.arange(y_test.shape[0]), math.ceil(0.3 * y_test.shape[0]), False)
indices_test = list(set(np.arange(y_test.shape[0])) - set(indices_val))

x_val = x_test[indices_val,]
x_test = x_test[indices_test]

y_val = y_test[indices_val,]
y_test = y_test[indices_test,]

#path_data = "D:/Data/LCZ42_Cities/"
path_data = "/data/lcz42_cities/"

validation_data_h5 = h5py.File(path_data + 'validation_data.h5', 'w')
validation_data_h5.create_dataset('x', data=x_val)
validation_data_h5.create_dataset('y', data=y_val)
validation_data_h5.close()

test_data_h5 = h5py.File(path_data + 'test_data.h5', 'w')
test_data_h5.create_dataset('x', data=x_test)
test_data_h5.create_dataset('y', data=y_test)
test_data_h5.close()