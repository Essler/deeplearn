import random
import sys
#import random
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow import keras
from keras.datasets import cifar10
from keras.preprocessing import image

# Get batch size as first and only argument
batch_size = int(sys.argv[1])

# (32x32px RGB images, labels 0-9)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 10 classes for CIFAR10, 0-9 for airplane, automobile, bird, cat, deer, dog, frog, horse, ship, & truck
num_classes = 10

# CIFAR10 contains 32x32px RGB images
img_rows, img_cols = 32, 32

# Categorize y labels, so filtering only for a single labeled number
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# y_test  =
# y_train =

# x_test  = tuple3(10K,32,32,3)
# x_train = tuple3(50k,32,32,3)

# Reshape x samples
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# !ERROR! ValueError: cannot reshape array of size 153600000 into shape (50000,32,32,1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#input_shape = (img_rows, img_cols, 1)

# Change data type to 32-bit floats and normalize colors by dividing by 255 (the largest 8-bit color value)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')


# Display 16 example images
# for i in range(16):
# 	tf.keras.preprocessing.image.array_to_img(x_train[i]).show()

### STEP 2 - Split Testing Set and Create Validation Set

## Hold-Out Validation
# 70/30 Hold-Out
split_pos = int(x_train.shape[0] * 0.30)
# First 30% is the validation dataset
x_validate = x_train[0:split_pos]
y_validate = y_train[0:split_pos]
# Last 70% is the testing dataset
x_train_h = x_train[split_pos:x_train.shape[0]]
y_train_h = y_train[split_pos:y_train.shape[0]]

print('Hold-Out')
print('  Test Size:     ', x_train_h.shape[0])
print('  Validate Size: ', x_validate.shape[0])
#('Test Size: ', 215040)
#('Validate Size: ', 92160)


## K-Fold Cross Validation
# 10-Fold
k = 10
# Shuffle dataset
# np.random.shuffle() is destructive, so work on copies of the dataset
x_train_k = x_train
y_train_k = y_train
np.random.shuffle(x_train_k)
np.random.shuffle(y_train_k)
# x_train_k = random.sample(x_train, x_train.shape[0])
# y_train_k = random.sample(y_train, y_train.shape[0])
# x_train_k2 = random.shuffle(x_train)
# y_train_k2 = random.shuffle(y_train)
# Split dataset into 10 groups
#(x_train_k, y_train_k, z1, z2, z3) = np.array_split((x_train_k, y_train_k), k) # ValueError: could not broadcast input array from shape (10000,32,32,3) into shape (10000)
# x_train_k = np.array_split(x_train_k, k) # ValueError: could not broadcast input array from shape (10000,32,32,3) into shape (10000)
#y_train_k = np.array_split(y_train_k, k) # ValueError: could not broadcast input array from shape (10000,32,32,3) into shape (10000)
x_train_k = np.dsplit(x_train_k, 0) # ZeroDivisionError: integer division or modulo by zero
x_train_k = np.dsplit(x_train_k, 1) # ValueError: could not broadcast input array from shape (10000,32,32,3) into shape (10000)
x_train_k = np.dsplit(x_train_k, 2) # ValueError: could not broadcast input array from shape (10000,32,32,3) into shape (10000)
x_train_k = np.dsplit(x_train_k, 3) # ValueError: array split does not result in an equal division



## Bootstrap Validation

# TODO: Need to find a way to have this sampling happen every iteration. Need N boostrap samples for N validation sets.
#num_iterations = x_train.shape[0] / batch_size
#indices = [random.randint(0,x_train.shape[0]-1) for x in range(batch_size)]
#indices = [random.randint(0,x_train.shape[0]-1) for x in range(x_train.shape[0])] # Same size as original data set (test), but is that per batch?
#print(indices)
#x_validate = 
#y_validate = 
#x_validate = np.random.choice(x_train, x_train.shape[0])
#(x_validate, y_validate) = 
print(np.random.choice((x_train, y_train), x_train.shape[0]))
print(type(np.random.choice((x_train, y_train), x_train.shape[0])))

print('Bootstrap')
print('  Test Size:     ', x_train.shape[0])
print('  Validate Size: ', x_validate.shape[0])
