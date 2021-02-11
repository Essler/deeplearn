import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

# 32x32px RGB images, labels 0-9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Change data type to 32-bit floats and normalize colors by dividing by 255 (the largest 8-bit color value)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')

# Display 16 example images
for i in range(16):
    image.array_to_img(x_train[i]).show()
