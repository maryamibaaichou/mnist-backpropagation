"""Download the public MNIST files and turn them into network inputs."""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.request

import numpy as np

# Yann LeCun's MNIST page, and the mirror that serves the same four files.
DATASET_PAGE = "http://yann.lecun.com/exdb/mnist/"
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)

# Published MD5 checksums for the official IDX gzip files.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def default_data_dir() -> str:
    """Directory for the downloaded IDX files. Override with MNIST_DIR."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.environ.get("MNIST_DIR", os.path.join(root, "data"))


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Digit k becomes a target vector with a 1 in position k and 0 elsewhere."""
    targets = np.zeros((n_classes, labels.shape[0]), dtype=float)
    targets[labels.astype(int), np.arange(labels.shape[0])] = 1.0
    return targets


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(name: str, destination: str) -> None:
    expected = FILES[name]
    temporary = destination + ".part"
    errors: list[str] = []
    for base in MIRRORS:
        url = base + name
        try:
            print(f"  downloading {name}")
            urllib.request.urlretrieve(url, temporary)
            if _md5(temporary) != expected:
                raise ValueError(f"checksum mismatch for {name}")
            os.replace(temporary, destination)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if os.path.exists(temporary):
                os.remove(temporary)
    raise RuntimeError(
        "Could not download the official MNIST file "
        f"{name}.\n" + "\n".join(errors)
    )


def ensure_mnist(data_dir: str) -> None:
    """Download any missing official file into data_dir and check its checksum."""
    os.makedirs(data_dir, exist_ok=True)
    for name, expected in FILES.items():
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and _md5(path) == expected:
            continue
        _download(name, path)


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


def load_mnist(data_dir: str | None = None) -> dict[str, np.ndarray]:
    """
    Load the train and test splits from the official MNIST files.

    Missing files are downloaded into data_dir. images_* stay 28x28.
    x_* are those pixels flattened to 784 numbers in [0, 1].
    y_* are one-hot targets with shape (10, n).
    labels_* are the original digits 0-9.
    """
    if data_dir is None:
        data_dir = default_data_dir()
    ensure_mnist(data_dir)

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
