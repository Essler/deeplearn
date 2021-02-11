import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

# 32x32px RGB images, labels 0-9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 10 classes for CIFAR10, 0-9 for airplane, automobile, bird, cat, deer, dog, frog, horse, ship, & truck
num_classes = 10

# CIFAR10 contains 32x32px RGB images
img_rows, img_cols = 32, 32

# Change data type to 32-bit floats and normalize colors by dividing by 255 (the largest 8-bit color value)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')

