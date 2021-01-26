import numpy as np
import h5py
import random
import math
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

def concatenate_cities_labels(cities_list):
    concatenated_mat = np.array([])
    for city in cities_list:
        name_tmp = path_data + city + ".h5"
        h5_tmp = h5py.File(name_tmp,'r')
        # If yet empty, initialize matrix with first city file
        if concatenated_mat.size == 0:
            concatenated_mat = np.array(h5_tmp['label'])
        # Otherwise: append existing matrix with new city file
        else:
            concatenated_mat = np.vstack((concatenated_mat, np.array(h5_tmp['label'])))
    return(concatenated_mat)

def concatenate_cities_patches(cities_list):
    concatenated_mat = np.array([])
    for city in cities_list:
        name_tmp = path_data + city + ".h5"
        h5_tmp = h5py.File(name_tmp,'r')
        # If yet empty, initialize matrix with first city file
        if concatenated_mat.size == 0:
            concatenated_mat = np.array(h5_tmp['sen2'])
        # Otherwise: append existing matrix with new city file
        else:
            concatenated_mat = np.vstack((concatenated_mat, np.array(h5_tmp['sen2'])))
    return(concatenated_mat)

def to_one_hot_majority(vote_mat, labels):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        # Retreive ith vote vector corresponding to ith patch
        vote_vec = vote_mat[i,:]
        for value in vote_vec:
            # Create vector of zeroes
            one = [0 for _ in range(len(labels))]
            # Set number at respective position (linked to respective class) to 1
            one[value - 1] = 1
            if one_hot_encoded:
                one_hot_encoded = list(map(add, one_hot_encoded, one))
            # initialize one_hot_encoded if list is yet empty
            else:
                one_hot_encoded = one
        # Find majority vote of empirical class distribution
        tmp_majority = np.where((one_hot_encoded == np.amax(one_hot_encoded)))
        # Create vector of zeroes
        one_hot_encoded = np.zeros(len(labels))
        # Set number at respective position (linked to respective class) to 1
        one_hot_encoded[tmp_majority] = 1
        # initialize one_hot_encoded_mat if list is yet empty
        if i == 0:
            one_hot_encoded_mat = one_hot_encoded
        else:
            one_hot_encoded_mat = np.vstack((one_hot_encoded_mat, one_hot_encoded))
    return(one_hot_encoded_mat)

test_split = 0.2
n_test_cities = math.ceil(test_split * len(city_list))

np.random.seed(13)
# Randomly sample 20% of cities as test cities
test_cities = list(np.array(city_list)[np.random.choice(len(city_list), int(n_test_cities), replace = False)])
# Take rest as train cities
train_cities = [city for city in city_list if city not in test_cities]

n_test_addons = math.ceil(test_split * len(addon_list))

np.random.seed(1313)
# Randomly sample 20% of addons as test cities
test_addons = list(np.array(addon_list)[np.random.choice(len(addon_list), int(n_test_addons), replace = False)])
# Take rest as train addons
train_addons = [addon for addon in addon_list if addon not in test_addons]

# Concatenate patches of train cities & addons
x_train = concatenate_cities_patches(train_cities + train_addons)
x_test = concatenate_cities_patches(test_cities + test_addons)

# Concatenate labels of individual train cities and transform to one-hot representation
y_train_cities = concatenate_cities_labels(train_cities).astype(int)
y_train_cities = to_one_hot_majority(y_train_cities, labels)
# Concatenate labels of individual train addons and transform to one-hot representation
y_train_addons = concatenate_cities_labels(train_addons).astype(int)
y_train_addons = to_one_hot_majority(y_train_addons, labels)

# Concatenate labels of train cities & addons
y_train = np.vstack((y_train_cities, y_train_addons))
y_train = y_train.astype(int)

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities = to_one_hot_majority(y_test_cities, labels)
# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons = to_one_hot_majority(y_test_addons, labels)

# Concatenate labels of test cities & addons
y_test = np.vstack((y_test_cities, y_test_addons))
y_test = y_test.astype(int)

train_data_h5 = h5py.File(path_data + 'train_data.h5', 'w')
train_data_h5.create_dataset('x', data=x_train)
train_data_h5.create_dataset('y', data=y_train)
train_data_h5.close()

test_data_h5 = h5py.File(path_data + 'test_data.h5', 'w')
test_data_h5.create_dataset('x', data=x_test)
test_data_h5.create_dataset('y', data=y_test)
test_data_h5.close()

def to_one_hot(vote_mat, labels):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        vote_vec = vote_mat[i,:]
        for value in vote_vec:
            one = [0 for _ in range(len(labels))]
            one[value - 1] = 1
            if one_hot_encoded:
                one_hot_encoded = list(map(add, one_hot_encoded, one))
            else: # initialize one_hot_encoded if list is yet empty
                one_hot_encoded = one
        if i == 0:
            one_hot_encoded_mat = np.asarray(one_hot_encoded)
        else:
            one_hot_encoded_mat = np.vstack((one_hot_encoded_mat, np.asarray(one_hot_encoded)))
    return(one_hot_encoded_mat)

# Concatenate labels of individual test cities and transform to one-hot representation
y_test_cities = concatenate_cities_labels(test_cities).astype(int)
y_test_cities = to_one_hot(y_test_cities, labels)

# Concatenate labels of individual test addons and transform to one-hot representation
y_test_addons = concatenate_cities_labels(test_addons).astype(int)
y_test_addons = to_one_hot(y_test_addons, labels)

y_test_cities = y_test_cities.astype(int)
y_test_cities = torch.from_numpy(y_test_cities)
y_test_cities = y_test_cities / 11

y_test_addons = y_test_addons.astype(int)
y_test_addons = torch.from_numpy(y_test_addons)
y_test_addons = y_test_addons / 9

entropies_test_cities = Categorical(probs = y_test_cities).entropy()
entropies_test_addons = Categorical(probs = y_test_addons).entropy()

# Concatenate entropies of test cities & addons
entropies_test = torch.cat((entropies_test_cities, entropies_test_addons), 0)

entropies_test_h5 = h5py.File(path_data + 'entropies_test.h5', 'w')
entropies_test_h5.create_dataset('entropies_test', data=entropies_test.numpy())
entropies_test_h5.close()