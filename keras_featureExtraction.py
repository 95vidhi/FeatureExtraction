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

def load_images(search_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    
    print('Going to read images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(search_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    
    print('Done reading.')
    
    return images, labels, img_names, cls

test_images, test_labels, test_img_names, test_cls = load_images('./dataset/test_set', 128, ['dogs', 'cats'])
train_images, train_labels, train_img_names, train_cls = load_images('./dataset/training_set', 128, ['dogs', 'cats'])

test_images, test_labels, test_img_names, test_cls = shuffle(test_images, test_labels, test_img_names, test_cls)
train_images, train_labels, train_img_names, train_cls = shuffle(train_images, train_labels, train_img_names, train_cls)

test_cls = test_cls.astype('S')
test_img_names = test_img_names.astype('S')

train_cls = train_cls.astype('S')
train_img_names = train_img_names.astype('S')


print('Starting test image prediction')
test_images = model.predict(test_images)
print('Done test image prediction')

print('Starting train image prediction')
train_images = model.predict(train_images)
print('Done train image prediction')

out_file = './features.h5'

print('Writing features to {}'.format(out_file))
with h5py.File(out_file, 'w') as hf:
    hf.create_dataset("train_images", data=train_images)
    hf.create_dataset("train_labels", data=train_labels)
    hf.create_dataset("train_img_names", data=train_img_names)
    hf.create_dataset("train_cls", data=train_cls)
    hf.create_dataset("test_images", data=test_images)
    hf.create_dataset("test_labels", data=test_labels)
    hf.create_dataset("test_img_names", data=test_img_names)
    hf.create_dataset("test_cls", data=test_cls)
print('Features written successfully')

print('Reading the file')
f = h5py.File(out_file, 'r')
print('Keys in the file :')
for k in f.keys():
    print(k)
    





