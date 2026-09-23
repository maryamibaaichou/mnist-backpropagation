"""
Train a feedforward network on MNIST.

  1. Data preparation
  2. Design the network architecture
  3. Initialize parameters
  4. Define the cost function
  5. Define the evaluation index
  6. Train the network
  7. Test the network
  8. Store the network parameters
  9. Use the trained network
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mnist_backprop.data import DATASET_PAGE, load_mnist
from mnist_backprop.network import NeuralNetwork, check_gradients

LAYER_SIZES = [784, 128, 10]
ALPHA = 1.0
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 3
MIN_IMPROVEMENT = 1e-4
SEED = 0
DEMO_COUNT = 20


def heading(step: int, title: str) -> None:
    print()
    print(f"Step {step}. {title}")
    print("-" * 60)


def parameter_count(layer_sizes: list[int]) -> int:
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += n_out * n_in + n_out
    return total


def minibatches(x, y, batch_size, rng):
    order = rng.permutation(x.shape[1])
    for start in range(0, x.shape[1], batch_size):
        chosen = order[start : start + batch_size]
        yield x[:, chosen], y[:, chosen]


def plot_cost(history: list[dict], path: str) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(epochs, [row["train_cost"] for row in history], label="train cost J")
    plt.plot(epochs, [row["test_cost"] for row in history], label="test cost J")
    plt.xlabel("pass through the training set")
    plt.ylabel("cost J")
    plt.title("Squared-error cost during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_application(images, labels, predictions, path: str) -> None:
    count = len(labels)
    columns = 5
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9, 1.7 * rows))
    for axis, image, true_digit, predicted in zip(axes.ravel(), images, labels, predictions):
        axis.imshow(image, cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        correct = int(predicted) == int(true_digit)
        axis.set_title(
            f"true {int(true_digit)}   pred {int(predicted)}",
            color="darkgreen" if correct else "firebrick",
            fontsize=9,
        )
    for axis in axes.ravel()[count:]:
        axis.axis("off")
    figure.suptitle("Read a handwritten digit", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def prepare_data() -> dict[str, np.ndarray]:
    """Step 1. Download MNIST, scale pixels, and encode labels."""
    heading(1, "Data preparation")
    print("MNIST: handwritten digits, Yann LeCun, Corinna Cortes,")
    print("and Christopher J. C. Burges.")
    print(DATASET_PAGE)
    data = load_mnist()
    print(f"Training images: {data['x_train'].shape[1]}")
    print(f"Test images:     {data['x_test'].shape[1]}")
    print("Each image is 28 by 28. Pixels are divided by 255 and flattened to 784 inputs.")
    print("Each label becomes a length-10 target. Digit 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].")
    print("Test images are held out from the weight updates.")
    return data


def describe_architecture() -> None:
    """Step 2. State the layer sizes and the number of parameters."""
    heading(2, "Design the network architecture")
    print("Input layer:  784 units, one per pixel.")
    print("Hidden layer: 128 units, sigmoid.")
    print("Output layer: 10 units, sigmoid, one per digit.")
    print("Each unit computes z = (weights · inputs) + bias, then a = sigmoid(z).")
    print(f"Trainable parameters: {parameter_count(LAYER_SIZES)}")


def initialize_parameters() -> NeuralNetwork:
    """Step 3. Draw the initial weights and set the learning rate."""
    heading(3, "Initialize parameters")
    network = NeuralNetwork(LAYER_SIZES, alpha=ALPHA, seed=SEED)
    print(f"Learning rate alpha = {ALPHA}")
    print("Weights are drawn from a normal distribution with scale 1/sqrt(inputs).")
    print("Biases start at 0.")
    for index, weight in enumerate(network.weights):
        print(
            f"  layer {index + 1}: weights {tuple(weight.shape)}, "
            f"biases {tuple(network.biases[index].shape)}"
        )
    return network


def define_cost(network: NeuralNetwork, x: np.ndarray, y: np.ndarray, label: int) -> None:
    """Step 4. Squared error between the network output and the target."""
    heading(4, "Define the cost function")
    print("Error of output unit j:  e_j = a_j - y_j")
    print("Cost of one image:        J = (1/2) * sum_j e_j^2")
    print("Cost of a batch:          the mean of the per-image costs.")
    _, output = network.forward(x[:, :1])
    sample_cost = network.cost(output, y[:, :1])
    print(f"Before training, image 0 (digit {label}) has J = {sample_cost:.4f}.")


def define_evaluation(network: NeuralNetwork, x: np.ndarray, labels: np.ndarray) -> None:
    """Step 5. Accuracy is the fraction of correct digit readings."""
    heading(5, "Define the evaluation index")
    print("Accuracy is the fraction of images whose largest output matches the digit.")
    print("Training minimizes J. The finished network is scored by accuracy.")
    initial = network.accuracy(x[:, :1000], labels[:1000])
    print(f"Accuracy on the first 1000 test images, before training: {initial:.2%}")


def train_network(network: NeuralNetwork, data: dict[str, np.ndarray], output_dir: str):
    """Step 6. Forward, cost, backpropagation, weight update, repeated."""
    heading(6, "Train the network")
    print("Comparing the analytic gradient with a numerical derivative.")
    gradient_error = check_gradients()
    print(f"  largest relative difference: {gradient_error:.3e}")
    if gradient_error > 1e-4:
        raise RuntimeError("the backward pass does not match the cost function")

    print(f"Each update uses {BATCH_SIZE} images:")
    print("  forward pass, cost J, backpropagation,")
    print("  W <- W - alpha * dJ/dW.")

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    labels_train, labels_test = data["labels_train"], data["labels_test"]
    rng = np.random.default_rng(SEED)
    history: list[dict] = []
    best_cost = float("inf")
    quiet_epochs = 0
    stopped_early = False

    for epoch in range(1, MAX_EPOCHS + 1):
        batch_costs = [
            network.train_batch(x_batch, y_batch)
            for x_batch, y_batch in minibatches(x_train, y_train, BATCH_SIZE, rng)
        ]
        _, test_output = network.forward(x_test)
        row = {
            "epoch": epoch,
            "train_cost": float(np.mean(batch_costs)),
            "test_cost": network.cost(test_output, y_test),
            "train_accuracy": network.accuracy(x_train, labels_train),
            "test_accuracy": network.accuracy(x_test, labels_test),
        }
        history.append(row)
        print(
            f"  pass {epoch:02d}   "
            f"train J={row['train_cost']:.4f}   test J={row['test_cost']:.4f}   "
            f"train acc={row['train_accuracy']:.2%}   test acc={row['test_accuracy']:.2%}"
        )
        if best_cost - row["train_cost"] > MIN_IMPROVEMENT:
            best_cost = row["train_cost"]
            quiet_epochs = 0
        else:
            quiet_epochs += 1
            if quiet_epochs >= PATIENCE:
                stopped_early = True
                print("  The cost has leveled off, so training stops.")
                break

    cost_plot = os.path.join(output_dir, "cost_curve.png")
    plot_cost(history, cost_plot)
    return history, gradient_error, stopped_early


def test_network(network: NeuralNetwork, data: dict[str, np.ndarray], history: list[dict]) -> dict:
    """Step 7. Score the network on images that did not update the weights."""
    heading(7, "Test the network")
    labels_test = data["labels_test"]
    mistakes = int(np.sum(network.predict(data["x_test"]) != labels_test))
    result = {
        "test_cost": history[-1]["test_cost"],
        "test_accuracy": history[-1]["test_accuracy"],
        "train_cost": history[-1]["train_cost"],
        "train_accuracy": history[-1]["train_accuracy"],
        "mistakes": mistakes,
    }
    print(f"Test cost J:   {result['test_cost']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.2%}")
    print(f"Mistakes:      {mistakes} out of {labels_test.shape[0]}")
    return result


def store_parameters(network: NeuralNetwork, output_dir: str) -> str:
    """Step 8. Write the learned weights and biases."""
    heading(8, "Store the network parameters")
    path = os.path.join(output_dir, "network_parameters.npz")
    network.save(path)
    print("Saved weights and biases to outputs/network_parameters.npz")
    return path


def use_trained_network(parameter_path: str, data: dict[str, np.ndarray], output_dir: str) -> str:
    """Step 9. Load the stored parameters and read new digits."""
    heading(9, "Use the trained network")
    restored = NeuralNetwork.load(parameter_path)
    labels = data["labels_test"][:DEMO_COUNT]
    predictions = restored.predict(data["x_test"][:, :DEMO_COUNT])
    figure_path = os.path.join(output_dir, "application_examples.png")
    plot_application(data["images_test"][:DEMO_COUNT], labels, predictions, figure_path)
    correct = int(np.sum(predictions == labels))
    print("Loaded outputs/network_parameters.npz")
    print(f"First {DEMO_COUNT} test images: {correct} correct.")
    for index in range(DEMO_COUNT):
        mark = "ok" if int(predictions[index]) == int(labels[index]) else "wrong"
        print(
            f"  image {index:02d}: true {int(labels[index])}, "
            f"read as {int(predictions[index])} ({mark})"
        )
    print("Figure: outputs/application_examples.png")
    return figure_path


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    data = prepare_data()
    describe_architecture()
    network = initialize_parameters()
    define_cost(network, data["x_train"], data["y_train"], int(data["labels_train"][0]))
    define_evaluation(network, data["x_test"], data["labels_test"])
    history, gradient_error, stopped_early = train_network(network, data, output_dir)
    result = test_network(network, data, history)
    parameter_path = store_parameters(network, output_dir)
    use_trained_network(parameter_path, data, output_dir)

    metrics = {
        "dataset": "MNIST",
        "dataset_source": DATASET_PAGE,
        "layer_sizes": LAYER_SIZES,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "parameter_count": parameter_count(LAYER_SIZES),
        "gradient_check_relative_error": gradient_error,
        "final_train_cost": result["train_cost"],
        "final_test_cost": result["test_cost"],
        "final_train_accuracy": result["train_accuracy"],
        "final_test_accuracy": result["test_accuracy"],
        "test_mistakes": result["mistakes"],
        "history": history,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print()
    print("Record: outputs/metrics.json")
    print("Cost curve: outputs/cost_curve.png")


if __name__ == "__main__":
    main()
