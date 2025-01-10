import keras.api.models as mod
import keras.api.layers as lay
import numpy as np
import matplotlib.pyplot as plt


pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for training

t = np.arange(1, N + 1)
np.random.seed(42)  
y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N) 

# Convert data to matrix format for training/testing
def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)


train = y[0:n_train]
test = y[n_train:]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

x_train = x_train.reshape((x_train.shape[0], step, 1))
x_test = x_test.reshape((x_test.shape[0], step, 1))

# Define the RNN model
model = mod.Sequential()
model.add(lay.SimpleRNN(units=32, input_shape=(step, 1), activation="relu"))
model.add(lay.Dense(units=1))

# Compile and train the model
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

# Make predictions for training and test data
predict_train = model.predict(x_train).flatten()
predict_test = model.predict(x_test).flatten()

predict_full = np.concatenate([predict_train, predict_test])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y, label="Original", color="blue", linestyle="-", linewidth=1)  
plt.plot(t[step:N-1], predict_full, label="Predict", color="red", linestyle="--", linewidth=2)  

ta
plt.axvline(x=n_train, color="purple", linestyle="-", linewidth=2)

plt.legend()
plt.tight_layout()
plt.show()
