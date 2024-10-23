import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.src.datasets import mnist
from keras.src.layers import Dense, Input
from numpy._typing import NDArray

(x_train, _), (x_test, _) = mnist.load_data()

x_train: NDArray = x_train.astype("float32") / 255
x_train: NDArray = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test: NDArray = x_test.astype("float32") / 255
x_test: NDArray = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

original_dim = 784
encoding_dim = 32

input_img = Input(shape=(original_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(original_dim, activation="sigmoid")(encoded)
input_encoded = Input(shape=(encoding_dim,))

# автоенкодер
autoencoder = Model(input_img, decoded)

# compile
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)

# representation
image_cols = 20
image_rows = 4
rec = autoencoder.predict(x_test)
plt.figure(layout="tight", figsize=(image_cols, image_rows))
for row in range(image_rows):
    for col in range(image_cols):
        idx = image_cols * row + col
        i = image_cols * 2 * row + col
        ax = plt.subplot(image_rows * 2, image_cols, i + 1)
        plt.imshow(x_test[idx].reshape((28, 28)))
        # plt.gray()
        ax.set_axis_off()

        i = image_cols * (2 * row + 1) + col
        ax = plt.subplot(image_rows * 2, image_cols, i + 1)
        plt.imshow(rec[idx].reshape((28, 28)))
        # plt.gray()
        ax.set_axis_off()

plt.savefig("4_8_4.png")
