from __future__ import absolute_import
import random
import numpy as np


def validate(model, test_data):
    y_hats = [(np.argmax(model(x)), np.argmax(y)) for (x, y) in test_data]
    valid_res = sum(int(x == y) for (x, y) in y_hats)

    return valid_res


def train(model, optimizer, training_data, test_data, epochs: int = 12, batch_size: int = 16):
    n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
        for batch in batches:
            optimizer.optimize(model, batch)
        if test_data:
            valid_res = validate(model, test_data)
            print(f'Epoch {j + 1}: {valid_res} / {n_test}, accuracy = {valid_res / n_test}')
        else:
            print(f'Epoch {j + 1} complete')


def encode_idx(idx):
    cd = np.zeros(10)
    cd[idx] = 1

    return cd


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from train.mnist_model import SGD, NNModel

    X, y_str = fetch_openml('mnist_784', version=1, return_X_y=True)
    OPTIMIZER = SGD(0.5)
    MODEL = NNModel()

    X_norm = X / 255.

    y_int = y_str.astype(int)

    y_v = np.array([encode_idx(i) for i in y_int])

    X_train, X_test = X_norm[:60000], X[60000:]
    y_train, y_test = y_v[:60000], y_v[60000:]

    training_dt = [(x.reshape(784, 1), y.reshape(10, 1)) for (x, y) in zip(X_train, y_train)]
    test_dt = [(x.reshape(784, 1), y.reshape(10, 1)) for (x, y) in zip(X_test, y_test)]

    train(MODEL, OPTIMIZER, training_dt, test_dt, epochs=16, batch_size=32)
