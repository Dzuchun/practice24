from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
)
from keras._tf_keras.keras.models import Sequential

# завантаження набору даних CIFAR -10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# нормалiзацiя пiксельних значень в дiапзонi [0; 1]
x_train, x_test = (
    x_train / 255.0,
    x_test / 255.0,
)

# Перетворення векторiв класiв у двiйковi матрицi класiв однократне(кодування)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Побудова архiтектури згорткової мережi
model = Sequential(
    [
        Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dense(10),
        Activation("softmax"),
    ]
)

# компiляцiя моделi
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Тренування фiтування() моделi
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# оцiнка моделi
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")
