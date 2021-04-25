import tensorflow as tf


def run():
    # Load VGG16 with pre-trained weights on ImageNet, including classification top.
    vgg16 = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # Output the model.
    vgg16.summary()

    # Freeze all pre-trained weights, enforcing training for only the inserted STN attention modules.
    for layer in vgg16.layers:
        layer.trainable = False


if __name__ == '__main__':
    run()
