import numpy as np
import h5py
import math
import torch
from torch.distributions import Categorical

from utils.transforms import to_one_hot_majority, to_one_hot, concatenate_cities_labels, concatenate_cities_patches

#split = "random"
split = "sophisticated"

overall_list = ['amsterdam', 'berlin', 'cologne', 'guangzhou', 'islamabad', 'jakarta',
                'london', 'losangeles', 'madrid', 'milan', 'moscow', 'mumbai', 'munich',
                'nairobi', 'paris', 'riodejaneiro', 'rome', 'zurich']

city_list = ['amsterdam', 'berlin', 'cologne', 'london', 'madrid',
             'milan', 'munich', 'paris', 'rome', 'zurich']

addon_list = ['guangzhou_addon', 'islamabad_addon', 'jakarta_addon', 'losangeles_addon', 'moscow_addon',
              'mumbai_addon', 'nairobi_addon', 'riodejaneiro_addon']

# Define labels
labels = np.arange(1,18)

path_data = "/data/lcz42_votes/data/"
#path_data = "D:/Data/LCZ_Votes/"

test_split = 0.2
n_test_cities = math.ceil(test_split * len(city_list))

if split == "random":
    np.random.seed(13)
    # Randomly sample 20% of cities as test cities
    test_cities = list(np.array(city_list)[np.random.choice(len(city_list), int(n_test_cities), replace=False)])
    # Take rest as train cities
    train_cities = [city for city in city_list if city not in test_cities]

    n_test_addons = math.ceil(test_split * len(addon_list))

    np.random.seed(1313)
    # Randomly sample 20% of addons as test cities
    test_addons = list(np.array(addon_list)[np.random.choice(len(addon_list), int(n_test_addons), replace=False)])
    # Take rest as train addons
    train_addons = [addon for addon in addon_list if addon not in test_addons]
elif split == "sophisticated":
    test_cities = ["madrid", "munich", "paris", "zurich"]
    # Take rest as train cities
    train_cities = [city for city in city_list if city not in test_cities]

    test_addons = ["losangeles_addon", "moscow_addon", "mumbai_addon", "riodejaneiro_addon"]
    # Take rest as train addons
    train_addons = [addon for addon in addon_list if addon not in test_addons]

# Concatenate patches of train cities & addons
x_train = concatenate_cities_patches(train_cities + train_addons)
x_test = concatenate_cities_patches(test_cities + test_addons)

# Concatenate labels of individual train cities and transform to one-hot representation
y_train_cities = concatenate_cities_labels(train_cities).astype(int)
y_train_cities = to_one_hot_majority(y_train_cities, labels, addon=False)
# Concatenate labels of individual train addons and transform to one-hot representation
y_train_addons = concatenate_cities_labels(train_addons).astype(int)
y_train_addons = to_one_hot_majority(y_train_addons, labels, addon=True)

# Concatenate labels of train cities & addons
y_train = np.vstack((y_train_cities, y_train_addons))
y_train = y_train.astype(int)

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities = to_one_hot_majority(y_test_cities, labels, addon=False)
# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons = to_one_hot_majority(y_test_addons, labels, addon=True)

# Concatenate labels of test cities & addons
y_test = np.vstack((y_test_cities, y_test_addons))
y_test = y_test.astype(int)

if split == "sophisticated":
    np.random.seed(42)
    indices_val = np.random.choice(np.arange(y_test.shape[0]), math.ceil(0.5 * y_test.shape[0]), False)
    indices_test = list(set(np.arange(y_test.shape[0])) - set(indices_val))

    x_val = x_test[indices_val,]
    x_test = x_test[indices_test]

    y_val = y_test[indices_val,]
    y_test = y_test[indices_test,]

# Save to file

train_data_h5 = h5py.File(path_data + 'train_data.h5', 'w')
train_data_h5.create_dataset('x', data=x_train)
train_data_h5.create_dataset('y', data=y_train)
train_data_h5.close()

if split == "sophisticated":
    validation_data_h5 = h5py.File(path_data + 'validation_data.h5', 'w')
    validation_data_h5.create_dataset('x', data=x_val)
    validation_data_h5.create_dataset('y', data=y_val)
    validation_data_h5.close()

test_data_h5 = h5py.File(path_data + 'test_data.h5', 'w')
test_data_h5.create_dataset('x', data=x_test)
test_data_h5.create_dataset('y', data=y_test)
test_data_h5.close()


############################################ Train entropies ###########################################################

# Concatenate labels of individual train cities and transform to one-hot representation
y_train_cities = concatenate_cities_labels(train_cities).astype(int)
y_train_cities = to_one_hot(y_train_cities, labels, addon=False)

# Concatenate labels of individual train addons and transform to one-hot representation
y_train_addons = concatenate_cities_labels(train_addons).astype(int)
y_train_addons = to_one_hot(y_train_addons, labels, addon=True)

y_train_cities = y_train_cities.astype(int)
y_train_cities = torch.from_numpy(y_train_cities)
y_train_cities = y_train_cities / y_train_cities.sum(axis=1, keepdims=True)

y_train_addons = y_train_addons.astype(int)
y_train_addons = torch.from_numpy(y_train_addons)
y_train_addons = y_train_addons / y_train_addons.sum(axis=1, keepdims=True)

entropies_train_cities = Categorical(probs = y_train_cities).entropy()
entropies_train_addons = Categorical(probs = y_train_addons).entropy()

# Concatenate entropies of train cities & addons
entropies_train = torch.cat((entropies_train_cities, entropies_train_addons), 0)

entropies_train_h5 = h5py.File(path_data + 'entropies_train.h5', 'w')
entropies_train_h5.create_dataset('entropies_train', data=entropies_train.numpy())
entropies_train_h5.close()

############################################ Test entropies ############################################################

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities = to_one_hot(y_test_cities, labels, addon=False)

# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons = to_one_hot(y_test_addons, labels, addon=True)

y_test_cities = y_test_cities.astype(int)
y_test_cities = torch.from_numpy(y_test_cities)
y_test_cities = y_test_cities / y_test_cities.sum(axis=1, keepdims=True)

y_test_addons = y_test_addons.astype(int)
y_test_addons = torch.from_numpy(y_test_addons)
y_test_addons = y_test_addons / y_test_addons.sum(axis=1, keepdims=True)

entropies_test_cities = Categorical(probs = y_test_cities).entropy()
entropies_test_addons = Categorical(probs = y_test_addons).entropy()

# Concatenate entropies of test cities & addons
entropies_test = torch.cat((entropies_test_cities, entropies_test_addons), 0)

if split == "sophisticated":
    entropies_test = entropies_test[indices_test, ]

entropies_test_h5 = h5py.File(path_data + 'entropies_test.h5', 'w')
entropies_test_h5.create_dataset('entropies_test', data=entropies_test.numpy())
entropies_test_h5.close()
