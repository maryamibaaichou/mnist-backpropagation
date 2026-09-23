"""
Forward pass, squared-error cost, and backpropagation.

Columns of an activation matrix are examples.

  e = a^L - y
  J = (1/2) sum_j e_j^2
  z^{l+1} = W^l a^l + b^l
  a^{l+1} = sigmoid(z^{l+1})
  dJ/dW^l = delta^{l+1} (a^l)^T
  W <- W - alpha * dJ/dW

Row i of W^l feeds unit i of the next layer, so
dJ/dW^l[i, j] = delta^{l+1}[i] * a^l[j].
The bias uses the same update with a constant input of 1.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def sigmoid_prime_from_activation(a: np.ndarray) -> np.ndarray:
    """f'(z) = f(z) (1 - f(z)), using the activation that was just computed."""
    return a * (1.0 - a)


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], alpha: float, seed: int = 0) -> None:
        if alpha <= 0:
            raise ValueError("learning rate alpha must be positive")
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input layer and an output layer")
        self.layer_sizes = list(layer_sizes)
        self.alpha = float(alpha)
        self.weights, self.biases = self._init_parameters(layer_sizes, seed)

    @staticmethod
    def _init_parameters(
        layer_sizes: list[int], seed: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng = np.random.default_rng(seed)
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = 1.0 / np.sqrt(n_in)
            weights.append(rng.normal(0.0, scale, size=(n_out, n_in)))
            biases.append(np.zeros(n_out, dtype=float))
        return weights, biases

    def forward(self, a: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Propagate activations from the input through every layer. Returns all of them and a^L."""
        activations = [a]
        for weight, bias in zip(self.weights, self.biases):
            z_next = weight @ activations[-1] + bias[:, None]
            activations.append(sigmoid(z_next))
        return activations, activations[-1]

    @staticmethod
    def output_error(a_L: np.ndarray, y: np.ndarray) -> np.ndarray:
        """e_j = a_j^L - y_j^L."""
        return a_L - y

    def cost(self, a_L: np.ndarray, y: np.ndarray) -> float:
        """Mean over the batch of J = (1/2) sum_j e_j^2."""
        error = self.output_error(a_L, y)
        per_example = 0.5 * np.sum(error ** 2, axis=0)
        return float(np.mean(per_example))

    def gradients(
        self, activations: list[np.ndarray], y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Backpropagate deltas, then dJ/dW^l = delta^{l+1} (a^l)^T.

        delta^L = (a^L - y) * sigmoid'(z^L)
        delta^l = sigmoid'(z^l) * (W^l)^T delta^{l+1}
        """
        a_L = activations[-1]
        delta = self.output_error(a_L, y) * sigmoid_prime_from_activation(a_L)
        batch_size = a_L.shape[1]

        grad_w_reversed: list[np.ndarray] = []
        grad_b_reversed: list[np.ndarray] = []
        for layer in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[layer]
            grad_w_reversed.append((delta @ a_prev.T) / batch_size)
            grad_b_reversed.append(np.mean(delta, axis=1))
            if layer > 0:
                delta = sigmoid_prime_from_activation(activations[layer]) * (
                    self.weights[layer].T @ delta
                )

        grad_w_reversed.reverse()
        grad_b_reversed.reverse()
        return grad_w_reversed, grad_b_reversed

    def update(
        self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]
    ) -> None:
        """W <- W - alpha * dJ/dW, and the same rule for biases."""
        for layer in range(len(self.weights)):
            self.weights[layer] -= self.alpha * grad_w[layer]
            self.biases[layer] -= self.alpha * grad_b[layer]

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        """Forward pass, cost, backpropagation, and one weight update. Returns J before the update."""
        activations, a_L = self.forward(x)
        batch_cost = self.cost(a_L, y)
        grad_w, grad_b = self.gradients(activations, y)
        self.update(grad_w, grad_b)
        return batch_cost

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Digit prediction is the output neuron with the largest activation."""
        _, a_L = self.forward(x)
        return np.argmax(a_L, axis=0)

    def accuracy(self, x: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == labels))

    def save(self, path: str) -> None:
        """Write every weight matrix and bias vector to disk."""
        payload = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int64),
            "alpha": np.array(self.alpha, dtype=np.float64),
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{index}"] = weight
            payload[f"b{index}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        """Rebuild the network from a file written by save()."""
        with np.load(path) as data:
            layer_sizes = [int(size) for size in data["layer_sizes"]]
            network = cls(layer_sizes, alpha=float(data["alpha"]), seed=0)
            for index in range(len(network.weights)):
                network.weights[index] = np.array(data[f"W{index}"], dtype=float)
                network.biases[index] = np.array(data[f"b{index}"], dtype=float)
        return network


def check_gradients(seed: int = 0, epsilon: float = 1e-5) -> float:
    """
    Compare a few analytic gradients with central differences.

    Returns the largest relative error. A correct backward pass is near 1e-7
    to 1e-6 for this epsilon.
    """
    rng = np.random.default_rng(seed)
    net = NeuralNetwork([6, 4, 3], alpha=0.1, seed=seed)
    x = rng.normal(size=(6, 5))
    y = rng.normal(size=(3, 5))
    activations, _ = net.forward(x)
    analytic_w, analytic_b = net.gradients(activations, y)

    def loss_with_current_params() -> float:
        _, a_L = net.forward(x)
        return net.cost(a_L, y)

    worst = 0.0
    probes = [(0, 0, 0), (0, 1, 2), (1, 2, 1)]
    for layer, row, col in probes:
        weight = net.weights[layer]
        original = weight[row, col]
        weight[row, col] = original + epsilon
        plus = loss_with_current_params()
        weight[row, col] = original - epsilon
        minus = loss_with_current_params()
        weight[row, col] = original
        numeric = (plus - minus) / (2.0 * epsilon)
        analytic = analytic_w[layer][row, col]
        worst = max(worst, abs(numeric - analytic) / (abs(numeric) + 1e-8))

    bias = net.biases[0]
    original = bias[1]
    bias[1] = original + epsilon
    plus = loss_with_current_params()
    bias[1] = original - epsilon
    minus = loss_with_current_params()
    bias[1] = original
    numeric = (plus - minus) / (2.0 * epsilon)
    analytic = analytic_b[0][1]
    worst = max(worst, abs(numeric - analytic) / (abs(numeric) + 1e-8))
    return worst
