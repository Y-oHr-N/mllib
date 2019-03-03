import logging
from time import perf_counter
from typing import Any Callable, Dict #noqa

import numpy as np
import pandas as pd # noqa
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .utils import is_estimator

optuna_is_installed = True

try:
    import optuna
except ImportError:
    optuna_is_installed = False


class Objective:
    """Objective function.

    Parameters
    ----------
    estimator
        Object to use to fit the data.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    X
        Training data.

    y
        Target variable.

    cv
        Cross-validation strategy.

    error_score
        Value to assign to the score if an error occurs in estimator fitting.

    fit_params
        Parameters passed to the ``fit`` method of the estimator.

    groups
        Group labels for the samples used while splitting the dataset into
        train/test set.

    return_train_score
        If True, training scores will be included.

    scoring
        String or callable to evaluate the predictions on the test data.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        X,
        y=None,
        cv=5,
        error_score='raise',
        fit_params=None,
        groups=None,
        return_train_score=False,
        scoring=None
    ):
        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.groups = groups
        self.param_distributions = param_distributions
        self.return_train_score = return_train_score
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
            error_score=self.error_score,
            fit_params=self.fit_params,
            groups=self.groups,
            return_train_score=self.return_train_score,
            scoring=self.scoring
        )

        for name, array in cv_results.items():
            if name in ['test_score', 'train_score']:
                for i, score in enumerate(array):
                    trial.set_user_attr(f'split{i}_{name}', score)

            trial.set_user_attr(f'mean_{name}', np.average(array))
            trial.set_user_attr(f'std_{name}', np.std(array))

        user_attrs = trial.user_attrs

        return - user_attrs['mean_test_score']


class TPESearchCV(BaseEstimator):
    """Hyper parameter search with cross-validation.

    Parameters
    ----------
    estimator
        Object to use to fit the data.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    cv
        Cross-validation strategy.

    error_score
        Value to assign to the score if an error occurs in estimator fitting.

    load_if_exists
        If True, the existing study is used in the case where a study named
        ``study_name`` already exists in the ``storage``.

    n_iter
        Number of trials.

    n_jobs
        Number of parallel jobs.

    random_state
        Seed of the pseudo random number generator.

    return_train_score
        If True, training scores will be included.

    refit
        If True, refit the estimator with the best found parameters.

    scoring
        String or callable to evaluate the predictions on the test data.

    storage
        Database URL.

    study_name
        name of the ``Study``.

    timeout
        Time limit in seconds for the search of appropriate models.

    verbose
        Verbosity level.

    Attributes
    ----------
    best_estimator_
        Estimator that was chosen by the search.

    n_splits_
        Number of cross-validation splits.

    refit_time_
        Time for refitting the best estimator.

    study_
        Study corresponds to the optimization task.

    Examples
    --------
    >>> from mllib.model_selection import TPESearchCV
    >>> from optuna.distributions import LogUniformDistribution
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> clf = SVC(gamma='auto')
    >>> param_distributions = {'C': LogUniformDistribution(1e-10, 1e+10)}
    >>> tpe_search = TPESearchCV(clf, param_distributions)
    >>> X, y = load_iris(return_X_y=True)
    >>> tpe_search.fit(X, y) # doctest: +ELLIPSIS
    TPESearchCV(...)
    >>> y_pred = tpe_search.predict(X)
    """

    # TODO: add a logic for pruning
    @property
    def _estimator_type(self):
        # () -> str
        return self.estimator._estimator_type

    @property
    def best_index_(self):
        # () -> int
        """Index which corresponds to the best candidate parameter setting.
        """

        self._check_is_fitted()

        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_params_(self):
        # () -> Dict[str, Any]
        """Parameters of the best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self):
        # () -> float
        """Mean cross-validated score of the best estimator.
        """

        self._check_is_fitted()

        return - self.best_value_

    @property
    def best_trial_(self):
        # () -> optuna.structs.FrozenTrial
        """Best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def best_value_(self):
        # () -> float
        """Best objective value in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def classes_(self):
        # () -> np.ndarray
        """Class labels.
        """

        self._check_is_fitted()

        return self.best_estimator_.classes_

    @property
    def n_iter_(self):
        # () -> int
        """Actual number of trials.
        """

        self._check_is_fitted()

        return len(self.study_.trials)

    @property
    def scorer_(self):
        # () -> Callable[..., float]
        """Scorer function.
        """

        self._check_is_fitted()

        return check_scoring(self.estimator, scoring=self.scoring)

    @property
    def decision_function(self):
        # (...) -> np.ndarray
        """Call decision_function on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        # (...) -> np.ndarray
        """Call inverse_transform on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        # (...) -> np.ndarray
        """Call predict on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        # (...) -> np.ndarray
        """Call predict_log_proba on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        # (...) -> np.ndarray
        """Call predict_proba on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        # (...) -> np.ndarray
        """Call transform on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.transform

    @property
    def trials_dataframe(self):
        # (...) -> pd.DataFrame
        """Call trials_dataframe on the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.trials_dataframe

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        error_score='raise',
        load_if_exists=False,
        n_iter=10,
        n_jobs=1,
        random_state=None,
        refit=True,
        return_train_score=False,
        scoring=None,
        storage=None,
        study_name=None,
        timeout=None,
        verbose=0
    ):
        # (...) -> None

        if not optuna_is_installed:
            raise ImportError('optuna is not installed')

        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self):
        # () -> None

        attributes = ['n_splits_', 'study_']

        if self.refit:
            attributes += ['best_estimator_', 'refit_time_']

        check_is_fitted(self, attributes)

    def _check_params(self):
        # () -> None

        if not is_estimator(self.estimator):
            raise ValueError(
                f'estimator must be a scikit-learn estimator'
            )

        if type(self.param_distributions) is not dict:
            raise ValueError(f'param_distributions must be a dictionary')

        for name, distribution in self.param_distributions.items():
            if not isinstance(
                distribution,
                optuna.distributions.BaseDistribution
            ):
                raise ValueError(
                    f'value of {name} must be a optuna distribution'
                )

    def _refit(self, X, y=None, **fit_params):
        # (np.ndarray, np.ndarray, Any) -> 'TPESearchCV'

        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.study_.best_params)

        start_time = perf_counter()

        self.best_estimator_.fit(X, y, **fit_params)

        self.refit_time_ = perf_counter() - start_time

        return self

    def _set_verbosity(self):
        # () -> None

        if self.verbose > 1:
            optuna.logging.set_verbosity(logging.DEBUG)
        elif self.verbose > 0:
            optuna.logging.set_verbosity(logging.INFO)
        else:
            optuna.logging.set_verbosity(logging.WARNING)

    def fit(self, X, y=None, groups=None, **fit_params):
        # (np.ndarray, np.ndarray, np.ndarray, Any) -> 'TPESearchCV'
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X
            Training data.

        y
            Target variable.

        groups
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self
            Return self.
        """

        self._check_params()
        self._set_verbosity()

        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        sampler = optuna.samplers.TPESampler(seed=seed)
        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=cv,
            error_score=self.error_score,
            fit_params=fit_params,
            groups=groups,
            return_train_score=self.return_train_score,
            scoring=self.scoring
        )

        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)
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

        if self.refit:
            self._refit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        # type: (np.ndarray, np.ndarray) -> float
        """Return the score on the given data.

        Parameters
        ----------
        X
            Data.

        y
            Target variable.

        Returns
        -------
        score
            Scaler score.
        """

        return self.scorer_(self.best_estimator_, X, y)
