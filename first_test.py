import numpy as np
import matplotlib.pyplot as plt


def test_matplotlib():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, marker='x')
    plt.show()


if __name__ == '__main__':
    test_matplotlib()
