import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
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

        cv_results = cross_validate(
            estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score=np.nan,
            fit_params=self.fit_params,
            return_train_score=True,
            scoring=self.scoring
        )

        for k, v in cv_results.items():
            trial.set_user_attr(f'mean_{k}', np.mean(v))
            trial.set_user_attr(f'std_{k}', np.std(v))

        return - np.mean(cv_results['test_score'])


class TPESearchCV(BaseEstimator):
    """Class for hyper parameter search with cross-validation.

    Examples
    --------
    >>> from mllib.model_selection import TPESearchCV
    >>> from optuna.distributions import LogUniformDistribution
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> clf = SVC(gamma='auto')
    >>> param_distributions = {'C': LogUniformDistribution(1e-10, 1e+10)}
    >>> X, y = load_iris(return_X_y=True)
    >>> tpe_search = TPESearchCV(clf, param_distributions)
    >>> tpe_search.fit(X, y) # doctest: +ELLIPSIS
    TPESearchCV(...)
    """

    @property
    def best_params_(self):
        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self):
        self._check_is_fitted()

        return - self.best_value_

    @property
    def best_trial_(self):
        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def best_value_(self):
        self._check_is_fitted()

        return self.study_.best_value

    @property
    def decision_function(self):
        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        self._check_is_fitted()

        return self.best_estimator_.transform

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        load_if_exists=False,
        n_iter=10,
        n_jobs=1,
        random_state=None,
        scoring=None,
        storage=None,
        study_name=None,
        timeout=None,
        verbose=False
    ):
        self.cv = cv
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self):
        check_is_fitted(self, ['best_estimator_', 'scorer_', 'study_'])

    def fit(self, X, y=None, **fit_params):
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        sampler = optuna.samplers.TPESampler(seed=seed)
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
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = optuna.create_study(
            load_if_exists=self.load_if_exists,
            sampler=sampler,
            storage=self.storage,
            study_name=self.study_name
        )

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
        self._check_is_fitted()

        return self.scorer_(self.best_estimator_, X, y)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
