import sys
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_batch():
    print("Execution starts when 'next()' is called for the first time.")
    try:
        # Backup dataset to restore later
        x_train_batch = x_train
        y_train_batch = y_train
        while True:
            try:
                print("Looping...")

                if x_train_batch.shape[0] < batch_size:
                    print("WARNING Exhausted Dataset")
                    raise StopIteration  # TODO: Hope this raise works...
                else:
                    x_batch = x_train_batch[0:batch_size]
                    y_batch = y_train_batch[0:batch_size]

                    x_train_batch = x_train_batch[batch_size:x_train_batch.shape[0]]
                    y_train_batch = y_train_batch[batch_size:y_train_batch.shape[0]]

                    yield x_batch, y_batch
            # Catch when no more items are left to generate
            except StopIteration:
                print("EXCEPTION StopIteration")

                data_remaining = x_train_batch.shape[0]
                print('x_train_batch.shape[0]: ', data_remaining)

                # Fill batch with as much from current dataset as possible
                x_batch = x_train_batch[0:data_remaining]
                y_batch = y_train_batch[0:data_remaining]

                # Reset and shuffle the dataset
                x_train_batch = x_train
                y_train_batch = y_train
                (x_train_batch, y_train_batch) = shuffle(x_train_batch, y_train_batch)

                # Fill the rest of the batch with the refreshed dataset
                x_batch = np.concatenate(x_batch, x_train_batch[0:batch_size-data_remaining]) # TODO: Hope this concat works...
                y_batch = np.concatenate(y_batch, y_train_batch[0:batch_size-data_remaining])
                x_train_batch = x_train_batch[batch_size-data_remaining:x_train_batch.shape[0]]
                y_train_batch = y_train_batch[batch_size-data_remaining:y_train_batch.shape[0]]

                yield x_batch, y_batch

                # "Restart" generator
                continue
            finally:
                print('While True; finally')

        print('While True; END')

    finally:
        print("get_batch(); finally")


# Get batch size as first and only argument
batch_size = int(sys.argv[1])

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

print("START")
batch_generator = get_batch()

# 800 batches of 64 = 51,200 images; more than the 50,000 in the original dataset
for x in range(800):
    (x_train_b, y_train_b) = next(batch_generator)
    print(x_train_b.shape[0])

batch_generator.close()
print("CLOSED")
