import os

import tensorflow as tf
from matplotlib import pyplot
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import np_utils, plot_model
# from matplotlib import pyplot
from keras.models import Model

print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # E tensorflow/stream_executor/cuda/cuda_driver.cc:328] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def step_1():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    nb_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    train_labels = np_utils.to_categorical(train_labels, nb_classes)
    test_labels = np_utils.to_categorical(test_labels, nb_classes)

    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    batch_size = 128
    epochs = 2
    pool_size = (2, 2)
    kernel_size = (5, 5)

    tf.keras.applications.MobileNet(
        input_shape=(32, 32, 3),
        alpha=1.0,  # default - a.k.a. 'resolution multiplier'
        depth_multiplier=1,  # default - a.k.a. 'resolution multiplier'
        dropout=0.001,  # default
        include_top=False,
        weights="imagenet",  # default
        pooling="avg"
    )

    from keras.layers import Input, Convolution2D, AveragePooling2D, Flatten, Dense

    inputs = Input(shape=input_shape)
    x = Convolution2D(6, kernel_size=kernel_size, strides=(1, 1), activation='tanh', input_shape=(28, 28, 1), padding='same')(inputs)
    x = AveragePooling2D(pool_size=pool_size, strides=(2, 2), padding='valid')(x)
    x = Convolution2D(16, kernel_size=kernel_size, strides=(1, 1), activation='tanh', padding='valid')(x)
    x = AveragePooling2D(pool_size=pool_size, strides=(2, 2), padding='valid')(x)
    x = Convolution2D(120, kernel_size=kernel_size, strides=(1, 1), activation='tanh', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(84, activation='tanh')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='mnist_lenet-5')

    model.summary()
    plot_model(model, "mnist_lenet-5.png", show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(test_images, test_labels))

    # Test model
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss: {:.4f}, Test accuracy {:.2f}%'.format(score[0], score[1] * 100))

    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    test_loss = history.history['val_loss']
    test_acc = history.history['val_accuracy']

    fig, axs = pyplot.subplots(2)
    fig.suptitle('LeNet-5')
    pyplot.xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlim(1, epochs)
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlim(1, epochs)
    axs[0].plot(range(1, epochs + 1), train_loss, 'r', label='Training')
    axs[0].plot(range(1, epochs + 1), test_loss, 'b', label='Testing')
    axs[1].plot(range(1, epochs + 1), train_acc, 'lightcoral', label='Training')
    axs[1].plot(range(1, epochs + 1), test_acc, 'cornflowerblue', label='Testing')
    axs[0].legend()
    axs[1].legend()
    pyplot.show()


def step_2():
    print('Training')


def step_3():
    print('Error Surface Study!')


if __name__ == '__main__':
    step_1()
    step_2()
    step_3()
