import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def test_matplotlib():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, marker='x')
    plt.rc('font', family='Verdana')
    plt.show()


def test_pandas():
    data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
            'Location': ['New York', 'Paris', 'Berlin', 'London'],
            'Age': [24, 13, 53, 33]}
    data_pandas = pd.DataFrame(data)
    display(data_pandas)
    print('Data from Pandas:\n{}'.format(data_pandas[data_pandas.Age > 30]))


def test_iris():
    iris_dataset = load_iris()
    print('Keys iris_dataset: \n{}'.format(iris_dataset.keys()))
    print('\nIris Dataset:\n')
    print(iris_dataset['DESCR'][:193] + '\n...')
    print('\nFirst 5 features: \n{}'.format(iris_dataset['data'][:5]))
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
    print('Shape array X_train: {}'.format(X_train.shape))
    print('Shape array y_train: {}'.format(y_train.shape))
    print('Shape array X_test: {}'.format(X_test.shape))
    print('Shape array y_test: {}'.format(y_test.shape))
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                      hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
    # plt.show(block=True)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    X_new = np.array([[5, 2.9, 1, 0.2]])
    print('Shape array X_new: {}'.format(X_new.shape))
    prediction = knn.predict(X_new)
    print('Predict: {}'.format(prediction))
    print('Predicting mark: {}'.format(iris_dataset['target_names'][prediction]))
    y_pred = knn.predict(X_test)
    print('Predicts for test array:\n {}'.format(y_pred))
    print('Correct on test array: {}'.format(np.mean(y_pred == y_test)))
    print('Score on test data: {}'.format(knn.score(X_test, y_test)))


if __name__ == '__main__':
    # test_matplotlib()
    # test_pandas()
    test_iris()
