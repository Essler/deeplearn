import tensorflow as tf


def run():
    vgg16 = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    vgg16.summary()


if __name__ == '__main__':
    run()
