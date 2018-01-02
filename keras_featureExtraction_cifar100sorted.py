#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:07:01 2017

@author: lakshay
"""

from __future__ import print_function

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
import cv2
import numpy as np
import os
import glob
from sklearn.utils import shuffle
import h5py
import pandas as pd

"""
dataframe = pd.read_csv('trainLabels.csv')
label_vals = dataframe.iloc[:, 1:2].values
labels = []
for val in label_vals:
    labels.append(val[0])
labels = np.array(labels).astype('S')

path = os.path.join('./', '*g')
files = glob.glob(path)
flbase = os.path.basename(files[0])
idx = np.int64(0)
for ch in flbase:
    if ch == '.' :
        break
    idx = idx*np.int64(10) + np.int64(ch)
idx = idx - np.int64(1)
"""

"""
search_path = "./data/cifar100/train"
subdir = os.listdir(search_path)
#subdir = subdir[1:]
files = []
for k in subdir:
    cur = os.path.join(search_path, k)
    subsubdir = os.listdir(cur)
    for val in subsubdir:
        curcur = os.path.join(cur, val, '*g')
        files += glob.glob(curcur)

"""

def load_images(search_path1, search_path2, image_size):
    images = []
    files = []
    labels = []
    subdir = os.listdir(search_path1)
    print('Going to read images from ', search_path1)
    for k in subdir:
        cur = os.path.join(search_path1, k)
        subsubdir = os.listdir(cur)
        for val in subsubdir:
            curcur = os.path.join(cur, val, '*g')
            ls = glob.glob(curcur)
            for fl in ls:
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_CUBIC)
                image = image.astype(np.float64)
                images.append(image)
                labels.append(k + '__' + val)
            files += ls
    print('Done reading.')
    subdir = os.listdir(search_path2)
    print('Going to read images from ', search_path2)
    for k in subdir:
        cur = os.path.join(search_path2, k)
        subsubdir = os.listdir(cur)
        for val in subsubdir:
            curcur = os.path.join(cur, val, '*g')
            ls = glob.glob(curcur)
            for fl in ls:
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_CUBIC)
                image = image.astype(np.float64)
                images.append(image)
                labels.append(k + '__' + val)
            files += ls
    images = np.array(images)
    files = np.array(files).astype('S')
    labels = np.array(labels).astype('S')
    print('Done reading.')
    return images, labels, files

train_images, train_labels, train_files = load_images('./image_datasets/data/cifar100/train', './image_datasets/data/cifar100/test', 150)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = x

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
for i, layer in enumerate(model.layers):
   print(i, layer.name)

print('Starting train image prediction')
train_images = model.predict(train_images)
print('Done train image prediction')

out_file = './features.h5'

print('Writing features to {}'.format(out_file))
with h5py.File(out_file, 'w') as hf:
    hf.create_dataset("train_images", data=train_images)
    hf.create_dataset("train_labels", data=train_labels)
print('Features written successfully')

print('Reading the file')
f = h5py.File(out_file, 'r')
print('Keys in the file :')
for k in f.keys():
    print(k)
    





