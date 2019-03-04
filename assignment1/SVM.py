import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC

from helpers import compute
from helpers import get_wine_data
from helpers import get_abalone_data
from helpers import plot_learning_curve
from helpers import plot_size_vs_accuracy
from helpers import plot_size_vs_time


def run():
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    classifier = SVC(C=10, kernel='linear')
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For Support Vector Machine With Linear Kernel"
    plot_learning_curve(classifier, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=4).show()

    classifier = SVC(C=10, kernel='linear')
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For Support Vector Machine With Linear Kernel"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.0, 1.), cv=cv, n_jobs=4).show()

    classifier = SVC(C=10, kernel='rbf', gamma=0.001)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For Support Vector Machine with RBF Kernel"
    plot_learning_curve(classifier, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=4).show()

    classifier = SVC(C=10, kernel='rbf', gamma=0.001)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For Support Vector Machine With Linear Kernel"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.0, 1.), cv=cv, n_jobs=4).show()


if __name__== "__main__":
    run()
