import numpy as np
import h5py
from operator import add
import torch

#path_data = "/data/lcz42_votes/data/"
path_data = "D:/Data/LCZ_Votes/"

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

def to_one_hot_majority(vote_mat, labels, addon):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        # Retreive ith vote vector corresponding to ith patch (12th entry is original vote) if City
        if addon == False:
            vote_vec = vote_mat[i,:11]
        else:
            vote_vec = vote_mat[i, :]
        for value in vote_vec:
            # Create vector of zeroes
            one = [0 for _ in range(len(labels))]
            # Set number at respective position (linked to respective class) to 1
            if value > 0:
                # Guarantees that only real votes get counted
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
        # Check for ties
        if len(tmp_majority[0]) == 1:
            # Set number at respective position (linked to respective class) to 1
            one_hot_encoded[tmp_majority] = 1
        else:
            # If tie, replace with original vote
            if addon == False:
                original_vote = vote_mat[i, 11]
                one_hot_encoded = np.zeros(len(labels))
                one_hot_encoded[original_vote - 1] = 1
            # For addon cities, ground truth is always class 7
            else:
                one_hot_encoded = np.zeros(len(labels))
                one_hot_encoded[6] = 1
        # initialize one_hot_encoded_mat if list is yet empty
        if i == 0:
            one_hot_encoded_mat = one_hot_encoded
        else:
            one_hot_encoded_mat = np.vstack((one_hot_encoded_mat, one_hot_encoded))
    return(one_hot_encoded_mat)

def to_one_hot(vote_mat, labels, addon):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        # Retreive ith vote vector corresponding to ith patch (12th entry is original vote) if City
        if addon == True:
            vote_vec = vote_mat[i,:]
        else:
            vote_vec = vote_mat[i, :11]
        for value in vote_vec:
            one = [0 for _ in range(len(labels))]
            if value > 0:
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