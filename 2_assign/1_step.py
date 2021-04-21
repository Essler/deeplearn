from keras.preprocessing import image
from keras.datasets import mnist
from keras.utils import np_utils

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="mnist.npz")

# 10 classes (numbers 0-9)
nb_classes = 10

# 28x28px single-channel (grayscale) images
img_rows, img_cols = 28, 28

train_labels = np_utils.to_categorical(train_labels, nb_classes)
test_labels = np_utils.to_categorical(test_labels, nb_classes)

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# Change data type to 32-bit floats and normalize colors by dividing by 255 (the largest 8-bit color value)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
print('train_images shape: ', train_images.shape)
print(train_images.shape[0], ' train samples')
print(test_images.shape[0], ' test samples')

# Display 16 example images
for i in range(16):
    image.array_to_img(train_images[i]).show()

batch_size = 128
epochs = 10
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())  # 2D to 1D
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(test_images, test_labels))

# Save the trained model
model.save('m_cnn_mnist.h5')

# Test model
from keras.models import load_model

test_model = load_model('m_cnn_mnist.h5')
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])
