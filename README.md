# MNIST network training

This project runs the nine network-training steps on the official MNIST digits: 60,000 training images and 10,000 test images of handwritten digits 0–9.

The pictures you can open live in `Documents/deep learning/mnist/images`. Training reads the official compressed files in that same folder:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## The nine steps

**1. Data preparation.** Each image is 28×28 grayscale. Pixel values are divided by 255 so they lie between 0 and 1, then flattened into 784 inputs. A label such as 3 becomes the target `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. The 10,000 test images are held out and never used to change the weights.

**2. Network architecture.** 784 inputs, one hidden layer of 128 sigmoid neurons, and 10 sigmoid outputs (one per digit). A neuron computes a weighted sum plus a bias, then passes it through the sigmoid.

**3. Initialize parameters.** 101,770 weights and biases. Weights start small and random. Biases start at 0. The learning rate is `alpha = 1.0`.

**4. Cost function.** For output neuron `j`, the error is `e_j = a_j - y_j`. The cost is `J = 1/2 * sum(e_j^2)`. On a batch, `J` is the average of the per-image costs. Training tries to make `J` smaller.

**5. Evaluation index.** Accuracy: the fraction of images where the largest output matches the true digit. Accuracy judges the network. The cost is what training minimizes. They are related, and they are not the same number.

**6. Train.** For each group of 128 images: forward pass, cost, backpropagation, then `weight <- weight - alpha * dJ/dweight`. The same rule updates the biases. This repeats for 20 passes over the training set. The gradient was checked against a numerical derivative before training (relative error about `2e-8`).

**7. Test.** After training, the held-out images give cost `J = 0.0392` and accuracy **95.49%** (451 mistakes out of 10,000).

**8. Store the parameters.** Weights and biases are saved in `outputs/network_parameters.npz`.

**9. Use the trained network.** The saved parameters are loaded again and asked to read digits. On the first 20 test images it got 19 right. The miss was a 5 read as a 6. See `outputs/application_examples.png`. The cost curve is `outputs/cost_curve.png`.

## Result of this run

| | Cost J | Accuracy |
|---|---:|---:|
| Before training (1,000 test images) | 1.29 on the first training image | 11.60% |
| After 20 passes, training set | 0.0384 | 95.91% |
| After 20 passes, test set | 0.0392 | 95.49% |
