import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test_matplotlib():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, marker='x')
    plt.show()


def test_pandas():
    data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
            'Location': ['New York', 'Paris', 'Berlin', 'London'],
            'Age': [24, 13, 53, 33]}
    data_pandas = pd.DataFrame(data)
    print('Data from Pandas:\n{}'.format(data_pandas[data_pandas.Age > 30]))


if __name__ == '__main__':
    # test_matplotlib()
    test_pandas()
