#!/usr/bin/env python3
"""Iris dataset analysis using PyTorch.
"""
from pathlib import Path

import numpy as np
import torch as T


class Batch:
    """Batch training data by randomly selecting items from training set.
    """

    def __init__(self, num_items, bat_size, seed=0):
        self.num_items = num_items
        self.bat_size = bat_size
        self.rnd = np.random.RandomState(seed)

    def next_batch(self):
        """Returns a numpy array of randomly selected indices.
        """
        return self.rnd.choice(self.num_items, self.bat_size, replace=False)


class Net(T.nn.Module):
    """Iris Dataset neural network with 7 node hidden layer.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = T.nn.Linear(4, 7)
        T.nn.init.xavier_uniform_(self.fc1.weight)
        T.nn.init.zeros_(self.fc1.bias)

        self.fc2 = T.nn.Linear(7, 3)
        T.nn.init.xavier_uniform_(self.fc2.weight)
        T.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):  # pylint: disable=arguments-differ
        z = T.tanh(self.fc1(x))
        z = self.fc2(z)
        return z


def accuracy(model, data_x, data_y):
    """Calculates the model accuracy as a percentage.
    """
    X = T.Tensor(data_x)
    Y = T.LongTensor(data_y)
    output = model(X)
    _, arg_maxes = T.max(output.data, dim=1)
    num_correct = T.sum(Y == arg_maxes)
    acc = num_correct * 100.0 / len(data_y)
    return acc.item()


def main():
    """Executes Iris Dataset demo using PyTorch.
    """
    print("\nBegin Iris Dataset with PyTorch demo\n")
    T.manual_seed(1)
    np.random.seed(1)

    print("loading Iris data into memory\n")

    train_file = Path("data/iris_train.csv")
    test_file = Path("data/iris_test.csv")

    train_x = np.loadtxt(
        train_file, usecols=range(0, 4), delimiter=",", skiprows=0, dtype=np.float32
    )
    train_y = np.loadtxt(
        train_file, usecols=[4], delimiter=",", skiprows=0, dtype=np.float32
    )
    test_x = np.loadtxt(
        test_file, usecols=range(0, 4), delimiter=",", skiprows=0, dtype=np.float32
    )
    test_y = np.loadtxt(
        test_file, usecols=[4], delimiter=",", skiprows=0, dtype=np.float32
    )

    net = Net()

    net = net.train()
    learning_rate = 0.01
    batch_size = 12
    max_iteration = 600
    n_items = len(train_x)

    loss_function = T.nn.CrossEntropyLoss()
    optimizer = T.optim.SGD(net.parameters(), lr=learning_rate)
    batcher = Batch(num_items=n_items, bat_size=batch_size)

    print("Start training")
    for i in range(0, max_iteration):
        if i > 0 and i % (max_iteration / 10) == 0:
            acc = accuracy(net, train_x, train_y)
            print(
                "iteration = {:4d}  loss = {:7.4f}  accuracy = {:0.2f}%".format(
                    i, loss_objective.item(), acc
                )
            )

        current_batch = batcher.next_batch()
        X = T.Tensor(train_x[current_batch])
        Y = T.LongTensor(train_y[current_batch])
        optimizer.zero_grad()
        output = net(X)
        loss_objective = loss_function(output, Y)
        loss_objective.backward()
        optimizer.step()
    print("Training complete\n")

    net = net.eval()
    acc = accuracy(net, test_x, test_y)
    print("Accuracy on test data = {:0.2f}%".format(acc))

    unknown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
    unknown = T.tensor(unknown)  # pylint: disable=not-callable
    logits = net(unknown)
    probs_t = T.softmax(logits, dim=1)
    probs = probs_t.detach().numpy()

    print("\nSetting inputs to:")
    for x in unknown[0]:
        print("%0.1f " % x, end="")
    print("\nPredicted: (setosa, versicolor, virginica)")
    for p in probs[0]:
        print("%0.4f " % p, end="")

    print("\n\nEnd Iris demo")


if __name__ == "__main__":
    main()
