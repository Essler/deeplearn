import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import expand_dims
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


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

    # Assuming "output of the feature maps after attention modules" means printing images from layer outputs.
    output_feature_maps_before_pooling(vgg16)


def output_feature_maps_before_pooling(model):
    # Indexes of the final conv layer of each block.
    indexes = [2, 5, 9, 13, 17]
    outputs = [model.layers[i + 1].output for i in indexes]
    model = Model(inputs=model.inputs, outputs=outputs)

    # Choose one example image, in this case from the 'gas pump' class.
    img = load_img('imagenette2/train/n03425413/n03425413_719.JPEG', target_size=(224,224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_maps = model.predict(img)
    plot_feature_maps(feature_maps)


def plot_feature_maps(feature_maps):
    square = 8
    for f_map in feature_maps:
        index = 1
        for _ in range(square):
            for _ in range(square):
                ax = plt.subplot(square, square, index)
                ax.set_xticks([])
                ax.set_yticks([])

                plt.imshow(f_map[0, :, :, index - 1], cmap='viridis')
                index += 1

    plt.show()


if __name__ == '__main__':
    run()
