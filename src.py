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
# енкодер
encoder = Model(input_img, encoded)
# декодер
decoder = Model(input_encoded, autoencoder.layers[-1](input_encoded))

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
display_size = 56
images_count = 15
enc = encoder.predict(x_test)
dec = decoder.predict(enc)
plt.figure(figsize=(20, 4))
for i in range(images_count):
    ax = plt.subplot(10, images_count, i + 1)
    plt.imshow(x_test[i].reshape((28, 28)))
    # plt.gray()
    ax.set_axis_off()

    ax = plt.subplot(10, 15, i + images_count + 1)
    plt.imshow(dec[i].reshape((28, 28)))
    # plt.gray()
    ax.set_axis_off()

plt.show()
