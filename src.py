import matplotlib.pyplot as plt
import numpy as np
from keras.api.layers import LSTM, Dense

t = np.linspace(0, 20, 1000, dtype=np.float32)
data = np.sin(t)

plt.plot(t, data, label="initial data")
plt.title("Sine")
# plt.show()

# створюємо набір даних
n_steps = 10
X = np.array([data[i : i + n_steps] for i in range(len(data) - n_steps)])
y = np.array(data[n_steps:])

# reshape X
X = X.reshape((X.shape[0], X.shape[1], 1))

from keras import Sequential

# створення моделі
model = Sequential(
    [
        LSTM(50, activation="relu"),  # LSTM doesn't seem to know `input_shape` argument
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")

# навчання моделі
model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
)

# прогнозуємо
predictions = 100
y_predict = np.array(
    [
        model.predict(
            data[
                len(data) - predictions - n_steps + i : len(data) - predictions + i
            ].reshape((1, n_steps, 1)),
            verbose="0",
        )[0][0]
        for i in range(predictions)
    ]
)

plt.plot(t[len(data) - predictions : len(data)], y_predict, label="predicted")
plt.legend()
plt.savefig("4_4.png")
