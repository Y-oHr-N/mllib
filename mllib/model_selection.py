import numpy as np
from optuna import create_study
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

__all__ = ['OptunaSearchCV']


class Objective:
    """Objective function.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        X,
        y=None,
        cv=5,
        scoring=None
    ):
        self.X = X
        self.y = y
        self.cv = cv
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring

    def __call__(self, trial):
        estimator = clone(self.estimator)
        params = {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }

        estimator.set_params(**params)

        scores = cross_val_score(
            estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score='raise',
            scoring=self.scoring
        )

        return - np.mean(scores)


class OptunaSearchCV(BaseEstimator):
    """Class for hyper parameter search with cross-validation.

    Examples
    --------
    >>> from mllib.model_selection import OptunaSearchCV
    >>> from optuna.distributions import LogUniformDistribution
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> clf = SVC(gamma='auto')
    >>> param_distributions = {'C': LogUniformDistribution(1e-10, 1e+10)}
    >>> X, y = load_iris(return_X_y=True)
    >>> optuna_search = OptunaSearchCV(clf, param_distributions).fit(X, y)
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        n_iter=10,
        n_jobs=1,
        random_state=None,
        scoring=None,
        verbose=False
    ):
        self.cv = cv
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.scoring = scoring

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring
        )

        self.sampler_ = TPESampler(seed=seed)
        self.study_ = create_study(sampler=self.sampler_)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_iter
        )

        self.best_params_ = self.study_.best_params
        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        check_is_fitted(self, ['best_estimator_', 'best_params_', 'scorer_'])

        X = check_array(X)

        return self.best_estimator_.predict(X)

    def score(self, X, y):
        check_is_fitted(self, ['best_estimator_', 'best_params_', 'scorer_'])

        X, y = check_X_y(X, y)

        return self.scorer_(self.best_estimator_, X, y)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
