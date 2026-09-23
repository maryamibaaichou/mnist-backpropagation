"""
Run the nine network-training steps on the official MNIST digits.

  1. Data preparation
  2. Design the network architecture
  3. Initialize parameters
  4. Define the cost function
  5. Define the evaluation index
  6. Train the network
  7. Test the network
  8. Store the network parameters
  9. Use the trained network for an application
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

from mnist_backprop.data import OFFICIAL_DIR, load_mnist
from mnist_backprop.network import NeuralNetwork, check_gradients

# Architecture chosen in step 2. One hidden layer is enough for these digits.
LAYER_SIZES = [784, 128, 10]
# Step size for the weight update w <- w - alpha * dJ/dw.
ALPHA = 1.0
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 3
MIN_IMPROVEMENT = 1e-4
SEED = 0


def heading(step: int, title: str) -> None:
    print()
    print(f"Step {step}. {title}")
    print("-" * 60)


def parameter_count(layer_sizes: list[int]) -> int:
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += n_out * n_in + n_out
    return total


def minibatches(x, y, labels, batch_size, rng):
    order = rng.permutation(x.shape[1])
    for start in range(0, x.shape[1], batch_size):
        chosen = order[start : start + batch_size]
        yield x[:, chosen], y[:, chosen], labels[chosen]


def plot_cost(history: list[dict], path: str) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(epochs, [row["train_cost"] for row in history], label="train cost J")
    plt.plot(epochs, [row["test_cost"] for row in history], label="test cost J")
    plt.xlabel("pass through the training set")
    plt.ylabel("cost J")
    plt.title("Squared-error cost while the weights are trained")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_application(images, labels, predictions, path: str) -> None:
    """Show the application: a picture goes in, a digit comes out."""
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
    figure.suptitle("Application: read a handwritten digit", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    heading(1, "Data preparation")
    print(f"Reading the official files in:\n  {OFFICIAL_DIR}")
    data = load_mnist(OFFICIAL_DIR)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    labels_train, labels_test = data["labels_train"], data["labels_test"]
    print(f"Training images: {x_train.shape[1]}")
    print(f"Test images:     {x_test.shape[1]}")
    print("Each image is 28 by 28. Pixel values 0-255 are divided by 255,")
    print("so every input is between 0 and 1, then flattened to 784 numbers.")
    print("Each label becomes a length-10 target. Digit 3 is")
    print("  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]")
    print("The test images are kept aside and are not used to change weights.")

    heading(2, "Design the network architecture")
    print("Input layer:  784 neurons, one per pixel.")
    print("Hidden layer: 128 neurons with a sigmoid activation.")
    print("Output layer: 10 neurons, also sigmoid, one per digit 0-9.")
    print("A neuron computes z = (weights · inputs) + bias, then a = sigmoid(z).")
    print(f"Trainable numbers: {parameter_count(LAYER_SIZES)}")

    heading(3, "Initialize parameters")
    network = NeuralNetwork(LAYER_SIZES, alpha=ALPHA, seed=SEED)
    print(f"Learning rate alpha = {ALPHA}")
    print("Weights start as small random numbers, scaled by 1/sqrt(inputs),")
    print("so the sigmoid is not stuck near 0 or 1 at the beginning.")
    print("Biases start at 0.")
    for index, weight in enumerate(network.weights):
        print(
            f"  layer {index + 1}: weight matrix {tuple(weight.shape)}, "
            f"bias vector {tuple(network.biases[index].shape)}"
        )

    heading(4, "Define the cost function")
    print("For one image, the error of output neuron j is e_j = a_j - y_j.")
    print("The cost is J = 1/2 * sum_j e_j^2.")
    print("On a group of images, J is the average of those per-image costs.")
    print("Training changes the weights to make J smaller.")
    sample_activations, sample_output = network.forward(x_train[:, :1])
    sample_cost = network.cost(sample_output, y_train[:, :1])
    print(
        f"Before any training, image 0 (digit {int(labels_train[0])}) "
        f"has cost J = {sample_cost:.4f}."
    )

    heading(5, "Define the evaluation index")
    print("The cost says how far the outputs are from the targets.")
    print("The evaluation index is accuracy: the fraction of images whose")
    print("largest output neuron matches the true digit.")
    print("Accuracy is how we judge the finished network. It is not the")
    print("quantity backpropagation differentiates.")
    initial_accuracy = network.accuracy(x_test[:, :1000], labels_test[:1000])
    print(f"Accuracy on the first 1000 test images, before training: {initial_accuracy:.2%}")

    heading(6, "Train the network")
    print("Checking that the computed gradient matches a numerical derivative...")
    gradient_error = check_gradients()
    print(f"  largest relative difference: {gradient_error:.3e}")
    if gradient_error > 1e-4:
        raise RuntimeError("the backward pass does not match the cost function")

    print("Each update uses a mini-batch of 128 images.")
    print("Forward pass, cost, backpropagation, then")
    print("  weight <- weight - alpha * (dJ / d weight).")
    print("That is the same update as the lecture, applied to 128 images at a time")
    print("so the full training set can be processed.")

    rng = np.random.default_rng(SEED)
    history: list[dict] = []
    best_cost = float("inf")
    quiet_epochs = 0
    stopped_early = False

    for epoch in range(1, MAX_EPOCHS + 1):
        batch_costs = [
            network.train_batch(x_batch, y_batch)
            for x_batch, y_batch, _ in minibatches(
                x_train, y_train, labels_train, BATCH_SIZE, rng
            )
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
                print("  The cost has stopped falling, so training stops.")
                break

    cost_plot = os.path.join(output_dir, "cost_curve.png")
    plot_cost(history, cost_plot)

    heading(7, "Test the network")
    test_accuracy = history[-1]["test_accuracy"]
    test_cost = history[-1]["test_cost"]
    predictions = network.predict(x_test)
    mistakes = int(np.sum(predictions != labels_test))
    print("The test images were not used in the weight updates.")
    print(f"Test cost J:  {test_cost:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Wrong digits: {mistakes} out of {labels_test.shape[0]}")

    heading(8, "Store the network parameters")
    parameter_path = os.path.join(output_dir, "network_parameters.npz")
    network.save(parameter_path)
    print(f"Saved weights and biases to:\n  {parameter_path}")

    heading(9, "Use the trained network for an application")
    print("The application reloads the stored parameters, then reads digits")
    print("it has not been shown as labeled answers during this step.")
    restored = NeuralNetwork.load(parameter_path)
    demo_count = 20
    demo_predictions = restored.predict(x_test[:, :demo_count])
    application_plot = os.path.join(output_dir, "application_examples.png")
    plot_application(
        data["images_test"][:demo_count],
        labels_test[:demo_count],
        demo_predictions,
        application_plot,
    )
    correct = int(np.sum(demo_predictions == labels_test[:demo_count]))
    print(f"On the first {demo_count} test images, it read {correct} correctly.")
    for index in range(demo_count):
        mark = "ok" if int(demo_predictions[index]) == int(labels_test[index]) else "wrong"
        print(
            f"  image {index:02d}: true {int(labels_test[index])}, "
            f"read as {int(demo_predictions[index])} ({mark})"
        )
    print(f"Picture of those readings:\n  {application_plot}")

    metrics = {
        "data_dir": OFFICIAL_DIR,
        "layer_sizes": LAYER_SIZES,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "parameter_count": parameter_count(LAYER_SIZES),
        "gradient_check_relative_error": gradient_error,
        "final_train_cost": history[-1]["train_cost"],
        "final_test_cost": test_cost,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": test_accuracy,
        "test_mistakes": mistakes,
        "history": history,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print()
    print(f"Record of the run:\n  {metrics_path}")
    print(f"Cost curve:\n  {cost_plot}")


if __name__ == "__main__":
    main()
