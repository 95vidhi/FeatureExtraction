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
search_path = "./train_new"
subdir = os.listdir(search_path)
#subdir = subdir[1:]
files = []
for k in subdir:
    cur = os.path.join(search_path, k, '*g')
    files += glob.glob(cur)
"""


def load_images(search_path, image_size):
    #images = []
    files = []
    labels = []
    subdir = os.listdir(search_path)
    print('Going to read images')
    for k in subdir:
        cur = os.path.join(search_path, k, '*g')
        ls = glob.glob(cur)
        for fl in ls:
            """image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_CUBIC)
            image = image.astype(np.float64)
            images.append(image)"""
            labels.append(k)
        files += ls
    #images = np.array(images)
    #files = np.array(files).astype('S')
    labels = np.array(labels).astype('S')
    print('Done reading.')
    return labels, files

train_labels, train_files = load_images('./image_datasets/train_new', 150)

"""k = np.array([[1,2,3], [12,3,3443], [2323,3432,2332], [1321,32,343422]])
v = np.zeros((4,3))
v[0] = k[0]
v[1] = k[1]
v[2] = k[2]"""

num_images = train_labels.shape[0]

train_images = np.zeros((num_images, 2048), dtype=np.float64)

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
image_size = 150
images = []
idx = 0
batch_size = 32
num_batches = num_images/batch_size
print("Number of batches : ", num_batches, ", each with batch size : ", batch_size)
batch_num = 0
for i in range(num_images):
    if len(images) == 32:
        batch_num = batch_num + 1
        print("Extracting features for batch number : ", batch_num)
        features = model.predict(np.array(images))
        for j in range(features.shape[0]):
            train_images[idx] = features[j]
            idx = idx + 1
        images = []
    image = cv2.imread(train_files[i])
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    images.append(image)
if len(images) != 0:
    features = model.predict(np.array(images), batch_size=len(images))
    for j in range(features.shape[0]):
        train_images[idx] = features[j]
        idx = idx + 1
print('Done train image prediction')

train_files = np.array(train_files).astype('S')
out_file = './features.h5'

print('Writing features to {}'.format(out_file))
with h5py.File(out_file, 'w') as hf:
    hf.create_dataset("train_images", data=train_images)
    hf.create_dataset("train_labels", data=train_labels)
    hf.create_dataset("train_files", data=train_files)
print('Features written successfully')

print('Reading the file')
f = h5py.File(out_file, 'r')
print('Keys in the file :')
for k in f.keys():
    print(k)
    





