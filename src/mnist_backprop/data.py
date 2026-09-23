"""Step 1. Read the official MNIST files and turn them into network inputs."""

from __future__ import annotations

import gzip
import os
import struct

import numpy as np

# Official gzip files in Documents/deep learning/mnist. Override with MNIST_DIR.
OFFICIAL_DIR = os.environ.get(
    "MNIST_DIR",
    "/Users/maryam/Documents/deep learning/mnist",
)

TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Digit k becomes a target vector with a 1 in position k and 0 elsewhere."""
    targets = np.zeros((n_classes, labels.shape[0]), dtype=float)
    targets[labels.astype(int), np.arange(labels.shape[0])] = 1.0
    return targets


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an MNIST image file (magic {magic})")
        pixels = np.frombuffer(handle.read(), dtype=np.uint8)
    if pixels.size != count * rows * cols:
        raise ValueError(f"{path} ended before all {count} images were read")
    return pixels.reshape(count, rows, cols)


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an MNIST label file (magic {magic})")
        labels = np.frombuffer(handle.read(), dtype=np.uint8).copy()
    if labels.size != count:
        raise ValueError(f"{path} ended before all {count} labels were read")
    return labels


def load_mnist(data_dir: str = OFFICIAL_DIR) -> dict[str, np.ndarray]:
    """
    Load train and test splits from the official IDX gzip files.

    images_* stay 28x28 so they can be looked at.
    x_* are those same pixels, flattened to 784 numbers in [0, 1].
    y_* are one-hot targets with shape (10, n).
    labels_* are the original digits 0-9.
    """
    required = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]
    missing = [name for name in required if not os.path.isfile(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Official MNIST files not found in {data_dir}: {', '.join(missing)}"
        )

    train_images = _read_images(os.path.join(data_dir, TRAIN_IMAGES))
    test_images = _read_images(os.path.join(data_dir, TEST_IMAGES))
    train_labels = _read_labels(os.path.join(data_dir, TRAIN_LABELS))
    test_labels = _read_labels(os.path.join(data_dir, TEST_LABELS))

    return {
        "images_train": train_images,
        "images_test": test_images,
        "x_train": train_images.reshape(train_images.shape[0], -1).T.astype(float) / 255.0,
        "x_test": test_images.reshape(test_images.shape[0], -1).T.astype(float) / 255.0,
        "y_train": one_hot(train_labels),
        "y_test": one_hot(test_labels),
        "labels_train": train_labels,
        "labels_test": test_labels,
    }
