# Pure NumPy neural network implementation — no deep learning frameworks used yet

import numpy as np
import os

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to your preprocessed data folder
x_path = os.path.join(script_dir, "preprocessed", "X_train.npy")
y_path = os.path.join(script_dir, "preprocessed", "y_train.npy")

# Load the data
x = np.load(x_path)
y = np.load(y_path)

# Flatten 3D IMU data into 2D for dense layers
x_flat = x.reshape(x.shape[0], -1)  # (11397, 512*9)

# --- Layer Classes ---
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

class ActivationSoftmaxLossCategoricalCrossentropy:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        return self.loss.calculate(self.activation.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# --- Network Architecture ---
dense1 = LayerDense(x_flat.shape[1], 256)
activation1 = ActivationReLU()

dense2 = LayerDense(256, 128)
activation2 = ActivationReLU()

num_classes = y.shape[1]
dense3 = LayerDense(128, 7)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# --- Adam Optimizer ---
class OptimizerAdam:
    def __init__(self, learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def increment_iteration(self):
        self.iterations += 1

# --- Training Loop ---
optimizer = OptimizerAdam(learning_rate=0.0005)
epochs = 200
max_grad_norm = 1.0  # gradient clipping threshold

for epoch in range(1, epochs + 1):
    # --- Forward Pass ---
    dense1.forward(x_flat)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y)

    predictions = np.argmax(loss_activation.activation.output, axis=1)
    y_classes = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y_classes)

    if epoch % 100 == 0 or epoch == 1:
        print(f"epoch {epoch} | loss {loss:.4f} | accuracy {accuracy:.4f}")

    # --- Backward Pass ---
    loss_activation.backward(loss_activation.activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # --- Gradient Clipping (added) ---
    for layer in [dense1, dense2, dense3]:
        np.clip(layer.dweights, -max_grad_norm, max_grad_norm, out=layer.dweights)
        np.clip(layer.dbiases, -max_grad_norm, max_grad_norm, out=layer.dbiases)

    # --- Update Parameters ---
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.increment_iteration()
