from typing import Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras import Model, Sequential
from keras.api.layers import Dropout
from keras.src.datasets import mnist
from keras.src.layers import Dense, Input, Reshape
from keras.src.metrics import CategoricalAccuracy
from keras.src.metrics.iou_metrics import confusion_matrix
from keras.src.utils import to_categorical
from numpy._typing import NDArray

dataset = mnist.load_data()
# normalize pixel values (so that they are between 0 and 1)
x_train = (dataset[0][0] - 127.5) / 127.5
x_test = (dataset[1][0] - 127.5) / 127.5
train_images: NDArray = dataset[0][0]

# here I'd like to train a AE on a dataset, and then use it's latent variables for class
# determination.

# AE settings
ori_shape: Tuple[int, int] = x_train.shape[1:]  # type:ignore
int_size = 64
lat_size = 32
cls_size = 512

encoder_filepath = "encoder.keras"
from os.path import isfile

encoder: Model = None  # type:ignore
if isfile(encoder_filepath):
    encoder = keras.models.load_model(encoder_filepath)  # type:ignore
else:
    # encoder
    def build_encoder(
        ori_shape: Tuple[int, int],
        int_size: int,
        lat_size: int,
        drp_rate: float,
    ) -> Model:
        inp_layer = Input(shape=ori_shape)
        rsh_layer = Reshape((np.prod(ori_shape),))(inp_layer)
        int_layer = Dense(int_size, activation=keras.activations.gelu)(rsh_layer)
        int_layer2 = Dropout(drp_rate)(int_layer)
        lat_layer = Dense(lat_size, activation=keras.activations.linear)(int_layer2)
        return Model(inputs=inp_layer, outputs=lat_layer, name="encoder")

    encoder = build_encoder(ori_shape, int_size, lat_size, 1.0 / int_size)

    # decoder
    def build_decoder(
        ori_shape: Tuple[int, int],
        int_size: int,
        lat_size: int,
        drp_rate: float,
    ) -> Model:
        inp_layer = Input(shape=(lat_size,))
        inp_layer2 = Dropout(drp_rate)(inp_layer)
        int_layer = Dense(int_size, activation=keras.activations.gelu)(inp_layer2)
        int_layer2 = Dropout(drp_rate)(int_layer)
        rsh_layer = Dense(np.prod(ori_shape), activation=keras.activations.tanh)(
            int_layer2
        )
        ori_layer = Reshape(ori_shape)(rsh_layer)
        return Model(inputs=inp_layer, outputs=ori_layer, name="decoder")

    decoder = build_decoder(ori_shape, int_size, lat_size, 1.0 / int_size)

    # autoencoder
    autoenc = Sequential(
        [
            encoder,
            decoder,
        ]
    )
    autoenc.compile(
        optimizer=keras.optimizers.Lion(),  # type:ignore
        loss=keras.losses.mean_squared_error,
    )
    autoenc.summary(expand_nested=True)
    autoenc.fit(
        x=x_train,
        y=x_train,
        batch_size=1000,
        epochs=100,
        validation_data=(x_test, x_test),
    )

    # creating visual comparison
    visual_sample = 10
    x = x_test[:visual_sample]
    y = autoenc.predict_on_batch(x)
    plt.figure(figsize=(20, 4))
    for i in range(visual_sample):
        # original
        ax = plt.subplot(2, visual_sample, i + 1)
        plt.imshow(x[i] * 127.5 + 127.5, cmap="gray")
        ax.set_axis_off()

        # reconstructed
        ax = plt.subplot(2, visual_sample, visual_sample + i + 1)
        plt.imshow(y[i] * 127.5 + 127.5, cmap="gray")
        ax.set_axis_off()

    plt.show()
    encoder.save(encoder_filepath, overwrite=True)


# classificator
def build_classifier(lat_size: int, classes: int, drp_rate: float) -> Model:
    inp_layer = Input(shape=(lat_size,))
    int_layer = Dense(cls_size, activation=keras.activations.leaky_relu)(inp_layer)
    int_layer2 = Dropout(drp_rate)(int_layer)
    out_layer = Dense(classes, activation=keras.activations.softmax)(int_layer2)
    return Model(inputs=inp_layer, outputs=out_layer, name="classifier_decoder")


classes = 10
classifier_decoder = build_classifier(lat_size, classes, 1.0 / lat_size)
classifier = Sequential(
    [
        encoder,
        classifier_decoder,
    ]
)
encoder.trainable = False
y_train = to_categorical(dataset[0][1], num_classes=classes)
y_test = to_categorical(dataset[1][1], num_classes=classes)
classifier.compile(
    optimizer=keras.optimizers.Nadam(),  # type:ignore
    loss=keras.losses.categorical_crossentropy,
    metrics=[
        CategoricalAccuracy(),
    ],
)
classifier.summary(expand_nested=True)
classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=1000,
    epochs=100,
    validation_data=(x_test, y_test),
)

# fine-tuning
encoder.trainable = True
classifier.compile(
    optimizer=keras.optimizers.AdamW(),  # type:ignore
    loss=keras.losses.categorical_crossentropy,
    metrics=[
        CategoricalAccuracy(),
    ],
)
classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=1000,
    epochs=100,
    validation_data=(x_test, y_test),
)

# I'd like to have a confusion matrix
y_pred = classifier.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
# Create confusion matrix and normalizes it over predicted (columns)
result: tensorflow.Tensor = confusion_matrix(
    y_test, y_pred, num_classes=classes
)  # type:ignore
print(result.numpy())  # type:ignore
