from keras.datasets import mnist
from keras.utils import np_utils, plot_model
from matplotlib import pyplot
from tensorflow.python.keras.models import Model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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

batch_size = 128
epochs = 10

from keras.layers import Input, Convolution2D, AveragePooling2D, Flatten, Dense

inputs = Input(shape=(28,28,1))
print(inputs.shape)
print(inputs.dtype)
# C1
x = Convolution2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1), padding='same')(inputs)
# S2
x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
# C3
x = Convolution2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid')(x)
# S4
x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
# C5
x = Convolution2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid')(x)
# Flatten
x = Flatten()(x)
# FC6
x = Dense(84, activation='tanh')(x)
# Output
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs, name='mnist_lenet-5')

# Default all layers as not trainable
for layer in model.layers:
    layer.trainable = False

# Then choose one, and only one, to train
# model.layers[0].trainable = True
# model.layers[1].trainable = True
# model.layers[2].trainable = True
# model.layers[3].trainable = True
# model.layers[4].trainable = True
model.layers[5].trainable = True
# model.layers[6].trainable = True

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(test_images, test_labels))

# Test model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss: {:.4f}, Test accuracy {:.2f}%'.format(score[0], score[1]*100))

train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

pyplot.suptitle('LeNet-5 1-Layer Train')
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.xlim(1, epochs)
pyplot.plot(range(1, epochs+1), train_acc, 'r', label='Train')
pyplot.plot(range(1, epochs+1), test_acc, 'b', label='Test')
pyplot.legend()
pyplot.show()
