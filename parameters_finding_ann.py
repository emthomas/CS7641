import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from helpers import get_wine_data
from helpers import get_abalone_data


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


models1 = {
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier()
}

params1 = {
    'AdaBoostClassifier':  {'n_estimators': np.arange(1, 20)},
    'GradientBoostingClassifier': {'n_estimators': np.arange(1, 20), 'learning_rate': 0.1 ** np.arange(1, 10)},
    'DecisionTreeClassifier': {'max_depth': np.arange(1, 20), 'min_samples_leaf': np.arange(1, 20)},
    'KNeighborsClassifier': {'n_neighbors': np.arange(1, 10)}
}

models2 = {
    'SVC': SVC()
}

params2 = {
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

models3 = {
    'MLPClassifier': MLPClassifier()
}

params3 = {
    'MLPClassifier': {
        'solver': ['lbfgs'],
        'max_iter': [1, 3, 5, 7, 9],
        'alpha': 10.0 ** -np.arange(1, 10),
        'hidden_layer_sizes': np.arange(10, 15),
        'random_state': [0,1]
    }
}

if __name__ == "__main__":
    X, y = get_wine_data()
    X1, y1 = get_abalone_data()

    # helper1 = EstimatorSelectionHelper(models1, params1)
    # helper1.fit(X, y, scoring='accuracy', n_jobs=4, cv=5)
    # results = (helper1.score_summary(sort_by='max_score'))
    # results.to_csv("out/wine_params_1.csv")

    # helper1 = EstimatorSelectionHelper(models1, params1)
    # helper1.fit(X1, y1, scoring='accuracy', n_jobs=4, cv=5)
    # results = (helper1.score_summary(sort_by='max_score'))
    # results.to_csv("out/abalone_params_1.csv")

    helper1 = EstimatorSelectionHelper(models2, params2)
    helper1.fit(X, y, scoring='accuracy', n_jobs=4, cv=5)
    results = (helper1.score_summary(sort_by='max_score'))
    results.to_csv("out/wine_params_2.csv")

    helper1 = EstimatorSelectionHelper(models2, params2)
    helper1.fit(X1, y1, scoring='accuracy', n_jobs=4, cv=5)
    results = (helper1.score_summary(sort_by='max_score'))
    results.to_csv("out/abalone_params_2.csv")

    # helper1 = EstimatorSelectionHelper(models3, params3)
    # helper1.fit(X, y, scoring='accuracy', n_jobs=4, cv=5)
    # results = (helper1.score_summary(sort_by='max_score'))
    # results.to_csv("out/wine_params_3.csv")

    # helper1 = EstimatorSelectionHelper(models3, params3)
    # helper1.fit(X1, y1, scoring='accuracy', n_jobs=4, cv=5)
    # results = (helper1.score_summary(sort_by='max_score'))
    # results.to_csv("out/abalone_params_3.csv")
