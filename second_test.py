import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display


def test_dataset_forge():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc=4)
    plt.xlabel('First feature.')
    plt.ylabel('Second feature.')
    print('Shape array: {}'.format(X.shape))
    plt.show()


def test_dataset_wave():
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel('Feature')
    plt.ylabel('Base variable')
    plt.show()


if __name__ == '__main__':
    # test_dataset_forge()
    test_dataset_wave()
