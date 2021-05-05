import numpy as np
import h5py
import math
import torch
from torch.distributions import Categorical

from utils.transforms import to_one_hot_majority, to_one_hot, concatenate_cities_labels, concatenate_cities_patches

# Define labels
labels = np.arange(1,18)

#path_data = "/data/lcz42_votes/data/"
path_data = "D:/Data/LCZ_Votes/"

alpha = 0.08

city_list = ['amsterdam', 'berlin', 'cologne', 'london', 'madrid',
             'milan', 'munich', 'paris', 'rome', 'zurich']

addon_list = ['guangzhou_addon', 'islamabad_addon', 'jakarta_addon', 'losangeles_addon', 'moscow_addon',
              'mumbai_addon', 'nairobi_addon', 'riodejaneiro_addon']

test_cities = ["madrid", "munich", "paris", "zurich"]
# Take rest as train cities
train_cities = [city for city in city_list if city not in test_cities]

test_addons = ["losangeles_addon", "moscow_addon", "mumbai_addon", "riodejaneiro_addon"]
# Take rest as train addons
train_addons = [addon for addon in addon_list if addon not in test_addons]

# Concatenate labels of individual test cities and transform to one-hot representation
y_train_cities = concatenate_cities_labels(train_cities).astype(int)
y_train_cities_majority = to_one_hot_majority(y_train_cities, labels, addon=False)
y_train_cities = to_one_hot(y_train_cities, labels, addon=False)
# Concatenate labels of individual test addons and transform to one-hot representation
y_train_addons = concatenate_cities_labels(train_addons).astype(int)
y_train_addons_majority = to_one_hot_majority(y_train_addons, labels, addon=True)
y_train_addons = to_one_hot(y_train_addons, labels, addon=True)

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities_majority = to_one_hot_majority(y_test_cities, labels, addon=False)
y_test_cities = to_one_hot(y_test_cities, labels, addon=False)
# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons_majority = to_one_hot_majority(y_test_addons, labels, addon=True)
y_test_addons = to_one_hot(y_test_addons, labels, addon=True)

# Concatenate labels of train cities & addons
y_train = np.vstack((y_train_cities, y_train_addons))
y_train = y_train.astype(int)

# Concatenate labels of train cities & addons after majority vote
y_train_majority = np.vstack((y_train_cities_majority, y_train_addons_majority))
y_train_majority = y_train_majority.astype(int)

# Concatenate labels of test cities & addons
y_test = np.vstack((y_test_cities, y_test_addons))
y_test = y_test.astype(int)

# Concatenate labels of test cities & addons after majority vote
y_test_majority = np.vstack((y_test_cities_majority, y_test_addons_majority))
y_test_majority = y_test_majority.astype(int)

indices_train = np.where(np.where(y_train_majority == np.amax(y_train_majority, 0))[1] + 1 < 11)[0]
# !!! Limit to 10 columns to receive distribution + entropy only w.r.t. urban classes !!!
y_train = y_train[indices_train, :10]

y_train_label_distributions = (y_train + alpha) / (y_train.sum(axis=1, keepdims=True) * (1 + alpha))

np.random.seed(42)
indices_val = np.random.choice(np.arange(y_test.shape[0]), math.ceil(0.5 * y_test.shape[0]), False)
indices_test = list(set(np.arange(y_test.shape[0])) - set(indices_val))

y_val = y_test[indices_val,]
y_val_majority = y_test_majority[indices_val,]
y_test = y_test[indices_test,]
y_test_majority = y_test_majority[indices_test,]

indices_val = np.where(np.where(y_val_majority == np.amax(y_val_majority, 0))[1] + 1 < 11)[0]
# !!! Limit to 10 columns to receive distribution + entropy only w.r.t. urban classes !!!
y_val = y_val[indices_val, :10]

y_val_label_distributions = (y_val + alpha) / (y_val.sum(axis=1, keepdims=True) * (1 + alpha))

indices_test = np.where(np.where(y_test_majority == np.amax(y_test_majority, 0))[1] + 1 < 11)[0]
# !!! Limit to 10 columns to receive distribution + entropy only w.r.t. urban classes !!!
y_test = y_test[indices_test, :10]

y_test_label_distributions = (y_test + alpha) / (y_test.sum(axis=1, keepdims=True) * (1 + alpha))

if alpha > 0:
    train_label_distributions_data_h5 = h5py.File(path_data + 'train_label_distributions_data' + '_alpha_' + alpha + '.h5', 'w')
    train_label_distributions_data_h5.create_dataset('train_label_distributions', data=y_train_label_distributions)
    train_label_distributions_data_h5.close()

    val_label_distributions_data_h5 = h5py.File(path_data + 'val_label_distributions_data' + '_alpha_' + alpha + '.h5', 'w')
    val_label_distributions_data_h5.create_dataset('val_label_distributions', data=y_val_label_distributions)
    val_label_distributions_data_h5.close()

    test_label_distributions_data_h5 = h5py.File(path_data + 'test_label_distributions_data' + '_alpha_' + alpha + '.h5', 'w')
    test_label_distributions_data_h5.create_dataset('test_label_distributions', data=y_test_label_distributions)
    test_label_distributions_data_h5.close()
else:
    train_label_distributions_data_h5 = h5py.File(path_data + 'train_label_distributions_data.h5', 'w')
    train_label_distributions_data_h5.create_dataset('train_label_distributions', data=y_train_label_distributions)
    train_label_distributions_data_h5.close()

    val_label_distributions_data_h5 = h5py.File(path_data + 'val_label_distributions_data.h5', 'w')
    val_label_distributions_data_h5.create_dataset('val_label_distributions', data=y_val_label_distributions)
    val_label_distributions_data_h5.close()

    test_label_distributions_data_h5 = h5py.File(path_data + 'test_label_distributions_data.h5', 'w')
    test_label_distributions_data_h5.create_dataset('test_label_distributions', data=y_test_label_distributions)
    test_label_distributions_data_h5.close()

