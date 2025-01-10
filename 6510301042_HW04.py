import keras.api.models as mod
import keras.api.layers as lay
import numpy as np
import matplotlib.pyplot as plt

# Generate data
pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for training

# Function to generate sine wave with random noise
t = np.arange(1, N + 1)
np.random.seed(42)  # For reproducibility
y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)  # Sine wave with noise

# Convert data to matrix format for training/testing
def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

# Split data into training and testing sets
train = y[0:n_train]
test = y[n_train:]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

# Reshape data for RNN input
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

# Combine predictions for plotting
predict_full = np.concatenate([predict_train, predict_test])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y, label="Original", color="blue", linestyle="-", linewidth=1)  # Original data
plt.plot(t[step:N-1], predict_full, label="Predict", color="red", linestyle="--", linewidth=2)  # Prediction line

# Add a vertical line to separate train and test data
plt.axvline(x=n_train, color="purple", linestyle="-", linewidth=2)

# Add labels, legend, and style
plt.legend()
plt.tight_layout()
plt.show()
