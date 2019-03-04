from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from helpers import get_wine_data
from helpers import get_abalone_data
from helpers import plot_learning_curve


def run():
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    classifier = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For Decision Tree"
    plot_learning_curve(classifier, title, X, y, ylim=(0.4, 0.6), cv=cv, n_jobs=4).show()

    classifier = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For Decision Tree"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.2, 0.4), cv=cv, n_jobs=4).show()


if __name__== "__main__":
    run()
