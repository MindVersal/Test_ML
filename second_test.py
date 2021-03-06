import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def test_dataset_forge():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc=4)
    plt.xlabel('First feature.')
    plt.ylabel('Second feature.')
    print('Shape array: {}'.format(X.shape))
    # fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    # for n_neighbors, ax in zip([1, 3, 9], axes):
    #     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    #     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    #     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    #     ax.set_title('Count neighbors:{}'.format(n_neighbors))
    #     ax.set_xlabel('Feature 0')
    #     ax.set_ylabel('Feature 1')
    # axes[0].legend(loc=3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{}'.format(clf.__class__.__name__))
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
    axes[0].legend()
    plt.show()


def test_dataset_wave():
    # first plot
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel('Feature')
    plt.ylabel('Base variable')
    # second plot
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)
    print('Score for testing: \n{}'.format(reg.predict(X_test)))
    print('R^2 on testing: {:.2f}'.format(reg.score(X_test, y_test)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
        ax.set_title('{} neighbors(s)\n train score: {:.2f} test score: {:.2f}'.format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
        ax.set_xlabel('Feature')
        ax.set_ylabel('Base variable')
    axes[0].legend(['Прогноз модели', 'Train data/answers', 'Test data/answers'], loc='best')
    plt.show()


def test_dataset_wave_linear_regression_basic():
    mglearn.plots.plot_linear_regression_wave()
    plt.show()


def test_dataset_wave_linear_regression_ols():
    X, y = mglearn.datasets.make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    print('lr.coef_: {}'.format(lr.coef_))
    print('lr.intercept_: {}'.format(lr.intercept_))
    print('Correct on train array: {:.2f}'.format(lr.score(X_train, y_train)))
    print('Correct on test array: {:.2f}'.format(lr.score(X_test, y_test)))


def test_dataset_cancer():
    cancer = load_breast_cancer()
    print('Keys in cancer: \n{}'.format(cancer.keys()))
    print('Shape data array in cancer: \n{}'.format(cancer.data.shape))
    print('Count primes for every classes in data cancer: \n{}'.format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    ))
    print('Feature names: \n{}'.format(cancer.feature_names))
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )
    # training_accuracy = []
    # test_accuracy = []
    # neighbors_settings = range(1, 11)
    # for n_neighbors in neighbors_settings:
    #     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    #     clf.fit(X_train, y_train)
    #     training_accuracy.append(clf.score(X_train, y_train))
    #     test_accuracy.append(clf.score(X_test, y_test))
    # plt.plot(neighbors_settings, training_accuracy, label='Correct in training')
    # plt.plot(neighbors_settings, test_accuracy, label='Correct in testing')
    # plt.ylabel('Correct')
    # plt.xlabel('Count neighbors')
    # plt.legend()
    logreg = LogisticRegression().fit(X_train, y_train)
    print('Correct on train array: {:.3f}'.format(logreg.score(X_train, y_train)))
    print('Correct on test array: {:.3f}'.format(logreg.score(X_test, y_test)))
    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print('Correct on train array (LogReg C=100): {:.3f}'.format(logreg100.score(X_train, y_train)))
    print('Correct on test array (LogReg C=100): {:.3f}'.format(logreg100.score(X_test, y_test)))
    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print('Correct on train array (LogReg C=0.01): {:.3f}'.format(logreg001.score(X_train, y_train)))
    print('Correct on test array (LogReg C=0.01): {:.3f}'.format(logreg001.score(X_test, y_test)))
    # plt.plot(logreg.coef_.T, 'o', label='C=1')
    # plt.plot(logreg100.coef_.T, '^', label='C=100')
    # plt.plot(logreg001.coef_.T, 'v', label='C=0.01')
    # plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    # plt.hlines(0, 0, cancer.data.shape[1])
    # plt.ylim(-5, 5)
    # plt.xlabel('Index coef')
    # plt.ylabel('Score coef')
    # plt.legend()
    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
        print('Correct on train array l1 with C={:.3f}: {:.2f}'.format(C, lr_l1.score(X_train, y_train)))
        print('Correct on test array l1 with C={:.3f}: {:.2f}'.format(C, lr_l1.score(X_test, y_test)))
        plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.xlabel('Index coef')
    plt.ylabel('Score coef')
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    plt.show()


def test_dataset_boston():
    boston = load_boston()
    print('Keys in boston: \n{}'.format(boston.keys()))
    print('Shape array boston: \n{}'.format(boston.data.shape))
    X, y = mglearn.datasets.load_extended_boston()
    print('Shape extended boston: \n{}'.format(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)
    print('Correct on train array: {:.2f}'.format(lr.score(X_train, y_train)))
    print('Correct on test array: {:.2f}'.format(lr.score(X_test, y_test)))
    print('Ridge testing:')
    ridge = Ridge().fit(X_train, y_train)
    print('Correct on train array: {:.2f}'.format(ridge.score(X_train, y_train)))
    print('Correct on test array: {:.2f}'.format(ridge.score(X_test, y_test)))
    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print('Correct on train array (Ridge alpha=10): {:.2f}'.format(ridge10.score(X_train, y_train)))
    print('Correct on test array (Ridge alpha=10): {:.2f}'.format(ridge10.score(X_test, y_test)))
    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    print('Correct on train array (Ridge alpha=0.1): {:.2f}'.format(ridge01.score(X_train, y_train)))
    print('Correct on test array (Ridge alpha=0.1): {:.2f}'.format(ridge01.score(X_test, y_test)))
    # plot Ridge
    # plt.plot(ridge.coef_, 's', label='Ridge regression alpha=1')
    # plt.plot(ridge10.coef_, '^', label='Ridge regression alpha=10')
    # plt.plot(ridge01.coef_, 'v', label='Ridge regression alpha=0.1')
    # plt.plot(lr.coef_, 'o', label='Linear regression')
    # plt.xlabel('Index coef')
    # plt.ylabel('Score coef')
    # plt.hlines(0, 0, len(lr.coef_))
    # plt.ylim(-25, 25)
    # plt.legend()
    print('Lasso Testing:')
    lasso = Lasso().fit(X_train, y_train)
    print('Correct on train array: {}'.format(lasso.score(X_train, y_train)))
    print('Correct on test array: {}'.format(lasso.score(X_test, y_test)))
    print('Count features: {}'.format(np.sum(lasso.coef_ != 0)))
    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print('Correct on train array (Lasso alpha=0.01): {}'.format(lasso001.score(X_train, y_train)))
    print('Correct on test array (Lasso alpha=0.01): {}'.format(lasso001.score(X_test, y_test)))
    print('Count features (Lasso alpha=0.01): {}'.format(np.sum(lasso001.coef_ != 0)))
    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print('Correct on train array (Lasso alpha=0.0001): {}'.format(lasso00001.score(X_train, y_train)))
    print('Correct on test array (Lasso alpha=0.0001): {}'.format(lasso00001.score(X_test, y_test)))
    print('Count features (Lasso alpha=0.0001): {}'.format(np.sum(lasso00001.coef_ != 0)))
    plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
    plt.plot(lasso001.coef_, '^', label='Lasso alpha=0.01')
    plt.plot(lasso00001.coef_, 'v', label='Lasso alpha=0.0001')
    plt.plot(ridge01.coef_, 'o', label='Ridge regression alpha=0.1')
    plt.legend(ncol=2, loc=(0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel('Index coef')
    plt.ylabel('Score coef')
    plt.show()


def test_dataset_blobs():
    X, y = make_blobs(random_state=42)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend(['Class 0', 'Class 1', 'Class 2'])
    linear_svm = LinearSVC().fit(X, y)
    print('Shape coef: {}'.format(linear_svm.coef_.shape))
    print('Shape const: {}'.format(linear_svm.intercept_.shape))
    line = np.linspace(-15, 15)
    mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 0', 'Line Class 1', 'Line Class 2'],
               loc=(1.01, 0.3))
    plt.show()


if __name__ == '__main__':
    # test_dataset_forge()
    # test_dataset_wave()
    # test_dataset_cancer()
    # test_dataset_boston()
    # test_dataset_wave_linear_regression_basic()
    # test_dataset_wave_linear_regression_ols()
    test_dataset_blobs()
