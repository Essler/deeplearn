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
epochs = 25

from keras.layers import Input, Convolution2D, AveragePooling2D, Flatten, Dense, GaussianNoise

inputs = Input(shape=(28,28,1))
print(inputs.shape)
print(inputs.dtype)
# C1
x = Convolution2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1), padding='same')(inputs)
# # Noise
# x = GaussianNoise(0.2)(x)
# S2
x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
# # Noise
# x = GaussianNoise(0.2)(x)
# C3
x = Convolution2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid')(x)
# Noise
x = GaussianNoise(0.2)(x)
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

model.summary()
plot_model(model, "mnist_lenet-5.png", show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(test_images, test_labels))

# Test model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss: {:.4f}, Test accuracy {:.2f}%'.format(score[0], score[1]*100))

train_loss = history.history['loss']
train_acc = history.history['accuracy']
test_loss = history.history['val_loss']
test_acc = history.history['val_accuracy']

fig, axs = pyplot.subplots(2)
fig.suptitle('LeNet-5 0.2 Noise L4')
pyplot.xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_xlim(1, epochs)
axs[1].set_ylabel('Accuracy')
axs[1].set_xlim(1, epochs)
axs[0].plot(range(1, epochs+1), train_loss, 'r', label='Training')
axs[0].plot(range(1, epochs+1), test_loss, 'b', label='Testing')
axs[1].plot(range(1, epochs+1), train_acc, 'lightcoral', label='Training')
axs[1].plot(range(1, epochs+1), test_acc, 'cornflowerblue', label='Testing')
axs[0].legend()
axs[1].legend()
pyplot.show()
