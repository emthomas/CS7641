from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from helpers import get_wine_data
from helpers import get_abalone_data
from helpers import plot_learning_curve


def run():
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    classifier = MLPClassifier(alpha=0.1, hidden_layer_sizes=13, max_iter=5, random_state=0, solver='lbfgs')
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For Neural Network"
    plot_learning_curve(classifier, title, X, y, ylim=(0.2, 0.6), cv=cv, n_jobs=4).show()

    classifier = MLPClassifier(alpha=0.1, hidden_layer_sizes=10, max_iter=9, random_state=0, solver='lbfgs')
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For Neural Network"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.1, 0.3), cv=cv, n_jobs=4).show()


if __name__== "__main__":
    run()
