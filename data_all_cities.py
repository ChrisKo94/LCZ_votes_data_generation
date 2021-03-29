import numpy as np
import h5py
import math
import glob
from utils.transforms import concatenate_cities_labels, concatenate_cities_patches

# Define labels
labels = np.arange(1,18)

path_data = "/data/lcz42_cities/"
#path_data = "D:/Data/LCZ42_Cities/"

evaluation_cities = ['amsterdam', 'berlin', 'cologne', 'guangzhou', 'islamabad', 'jakarta',
                     'london', 'losangeles', 'madrid', 'milan', 'moscow', 'mumbai', 'munich',
                     'nairobi', 'paris', 'riodejaneiro', 'rome', 'zurich']

overall_list = ['amsterdam', 'beijing', 'berlin', 'bogota', 'buenosaires', 'cairo', 'capetown', 'caracas',
                'changsha', 'chicago', 'cologne', 'dhaka', 'dongying', 'guangzhou', 'hongkong', 'islamabad',
                'istanbul', 'jakarta', 'kyoto', 'lima', 'lisbon', 'london', 'losangeles', 'madrid', 'melbourne',
                'milan', 'moscow', 'mumbai', 'munich', 'nairobi', 'nanjing', 'newyork', 'orangitown', 'paris',
                'philadelphia', 'qingdao', 'quezon', 'riodejaneiro', 'rome', 'salvador', 'sanfrancisco',
                'santiago', 'saopaulo', 'shanghai' , 'shenzhen', 'sydney', 'tehran', 'tokyo', 'vancouver',
                'washingtondc', 'wuhan', 'zurich']

train_cities = [city for city in overall_list if city not in evaluation_cities]

x_train = concatenate_cities_patches(train_cities)
y_train = concatenate_cities_labels(train_cities).astype(int)

train_data_h5 = h5py.File(path_data + 'train_data.h5', 'w')
train_data_h5.create_dataset('x', data=x_train)
train_data_h5.create_dataset('y', data=y_train)
train_data_h5.close()