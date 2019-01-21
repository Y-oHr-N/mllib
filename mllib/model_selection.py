import numpy as np
from optuna import create_study
from optuna.logging import disable_default_handler, enable_default_handler
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


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
        fit_params=None,
        scoring=None
    ):
        self.X = X
        self.y = y
        self.cv = cv
        self.estimator = estimator
        self.fit_params = fit_params
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
            error_score=np.nan,
            fit_params=self.fit_params,
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
    >>> optuna_search = OptunaSearchCV(clf, param_distributions)
    >>> optuna_search.fit(X, y) # doctest: +ELLIPSIS
    OptunaSearchCV(...)
    """

    @property
    def best_params_(self):
        return self.study_.best_params

    @property
    def best_score_(self):
        return - self.best_value_

    @property
    def best_trial_(self):
        return self.study_.best_trial

    @property
    def best_value_(self):
        return self.study_.best_value

    @property
    def decision_function(self):
        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        return self.best_estimator_.transform

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        n_iter=10,
        n_jobs=1,
        random_state=None,
        scoring=None,
        timeout=None,
        verbose=False
    ):
        self.cv = cv
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.scoring = scoring
        self.timeout = timeout
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        sampler = TPESampler(seed=seed)
        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=self.cv,
            fit_params=fit_params,
            scoring=self.scoring
        )

        if self.verbose:
            enable_default_handler()
        else:
            disable_default_handler()

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = create_study(sampler=sampler)

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_iter,
            timeout=self.timeout
        )

        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.study_.best_params)
        self.best_estimator_.fit(X, y)

        return self

    def score(self, X, y):
        check_is_fitted(self, ['best_estimator_', 'scorer_', 'study_'])

        return self.scorer_(self.best_estimator_, X, y)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
