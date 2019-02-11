from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from helpers import get_wine_data
from helpers import get_abalone_data
from helpers import plot_learning_curve


def run():
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, splitter='random')
    classifier = AdaBoostClassifier(base_estimator=dt, random_state=0, n_estimators=3)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Wine - Validation Curve For AdaBoostClassifier"
    plot_learning_curve(classifier, title, X, y, ylim=(0.4, 0.6), cv=cv, n_jobs=4).show()

    dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=3, splitter='random')
    classifier = AdaBoostClassifier(base_estimator=dt, random_state=0, n_estimators=15)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Abalone - Validation Curve For AdaBoostClassifier"
    plot_learning_curve(classifier, title, X1, y1, ylim=(0.1, 0.3), cv=cv, n_jobs=4).show()


if __name__== "__main__":
    run()
