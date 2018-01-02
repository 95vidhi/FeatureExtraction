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

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

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

def load_images(search_path, image_size):
    images = []
    dataframe = pd.read_csv('./dataset/trainLabels.csv')
    label_vals = dataframe.iloc[:, 1:2].values
    labels = []
    path = os.path.join(search_path, '*g')
    files = glob.glob(path)
    print('Going to read images')
    for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            images.append(image)
            flbase = os.path.basename(fl)
            idx = np.int64(0)
            for ch in flbase:
                if ch == '.' :
                    break
                idx = idx*np.int64(10) + np.int64(ch)
            idx = idx - np.int64(1)
            labels.append(label_vals[idx][0])
    images = np.array(images)
    labels = np.array(labels).astype('S')
    print('Done reading.')
    return images, labels

train_images, train_labels = load_images('./dataset/train', 32)

train_images, train_labels = shuffle(train_images, train_labels)

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
    





