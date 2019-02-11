from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from helpers import get_wine_data
from helpers import get_abalone_data
from helpers import plot_learning_curve


def run():
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    classifier = KNeighborsClassifier(n_neighbors=2)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For KNeighborsClassifier"
    plot_learning_curve(classifier, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=4).show()

    classifier = KNeighborsClassifier(n_neighbors=2)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For KNeighborsClassifier"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.0, 1.), cv=cv, n_jobs=4).show()


if __name__ == "__main__":
    run()
