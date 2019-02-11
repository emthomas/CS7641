import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import time
import numpy as np
import seaborn as sns


def get_wine_data():
    label = "quality"
    dataset = pd.read_csv('./data/wine/winequality-white.csv', sep=",")
    X = dataset.drop(label, 1)
    y = dataset[label]
    return X, y


def get_abalone_data():
    label = "rings"
    dataset = pd.read_csv('./data/abalone/abalone_data.csv', sep=',')
    mapping = {'M': 1, 'I': 2, 'F': 3}
    dataset.sex = dataset.sex.replace(mapping)
    X = dataset.drop(label, 1)
    y = dataset[label]
    return X, y


def compute(X, y, classifier, x_values):
    train = []
    test = []
    trainSize = []
    trainTime = []
    for i in x_values:
        test_size = 1-(i/100.0)
        x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=test_size)

        start = time.time()
        classifier.fit(x_train, y_train)
        end = time.time()

        train_predict = classifier.predict(x_train).astype(int)
        test_predict = classifier.predict(x_test).astype(int)
        train.append(accuracy_score(y_train, train_predict))
        test.append(accuracy_score(y_test, test_predict))
        trainSize.append(i/100.0)
        trainTime.append(end-start)
    return test, train, trainSize, trainTime


def plot_size_vs_accuracy(param_range, testData, trainingData, title):
    plt.plot(param_range, testData, label="Test", color="red")
    plt.plot(param_range, trainingData, label="Train", color="blue")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()


def plot_size_vs_time(size, values, title):
    plt.plot(size, values, label="Test", color="red")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Training Time in Seconds")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_distribution(dataset, label, title, x_label, y_label, bins):
    # matplotlib histogram
    plt.hist(dataset[label], color='blue', edgecolor='black', bins=bins)

    # seaborn histogram
    sns.distplot(dataset[label], hist=True, kde=False, bins=bins, color='blue', hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt


