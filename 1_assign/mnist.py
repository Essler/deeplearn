#import tensorflow as tf
#from tensorflow import keras
from keras.datasets import mnist
from keras.utils import np_utils
# Use MNIST, shuffle and split into train and test
# Images, Labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 10 classes in MNIST, digits 0-9
nb_classes = 10
# MNIST dataset contains 28x28px images with one channel (black & white)
img_rows, img_cols = 28, 28

# Categorize y labels, so filtering only for a single labeled number
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Reshape x samples
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Change data type to 32-bit floats and normalize gray intensity by dividing by 255 (the largest 8-bit intensity)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')
