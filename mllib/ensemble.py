import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor
)
from sklearn.ensemble import BaseEnsemble
from sklearn.utils import (
    check_X_y,
    check_array,
    check_consistent_length,
    check_random_state,
    column_or_1d
)
from sklearn.utils.validation import check_is_fitted

from .utils import is_estimator


class BaseRandomSeedAveraging(BaseEnsemble, ABC):
    # TODO: add a n_jobs parameter
    # TODO: add a verbose parameter

    @property
    def _check_params(self):
        return self._validate_estimator

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators
        )

        self.random_state = random_state

    def _check_is_fitted(self):
        check_is_fitted(self, 'estimators_')

    def fit(self, X, y=None, **fit_params):
        self._check_params()

        random_state = check_random_state(self.random_state)

        self.estimators_ = []

        for _ in range(self.n_estimators):
            e = self._make_estimator(random_state=random_state)

            e.fit(X, y, **fit_params)

        return self

    @abstractmethod
    def predict(self, X):
        pass


class RandomSeedAveragingClassifier(BaseRandomSeedAveraging, ClassifierMixin):
    """Random seed averaging classifier.

    examples
    --------
    >>> from mllib.ensemble import RandomSeedAveragingClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> est = RandomForestClassifier(n_estimators=10)
    >>> clf = RandomSeedAveragingClassifier(est)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y) # doctest: +ELLIPSIS
    RandomSeedAveragingClassifier(...)
    >>> y_pred = clf.predict(X)
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )

    def _check_params(self):
        super()._check_params()

        if not is_classifier(self.base_estimator):
            raise ValueError(f'base_estimator must be a classifier')

    def predict(self, X):
        self._check_is_fitted()

        predictions = np.asarray([e.predict(X) for e in self.estimators_]).T

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            mode, _ = stats.mode(predictions, axis=1)

        return np.ravel(mode)


class RandomSeedAveragingRegressor(BaseRandomSeedAveraging, RegressorMixin):
    """Random seed averaging regressor.

    examples
    --------
    >>> from mllib.ensemble import RandomSeedAveragingRegressor
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> est = RandomForestRegressor(n_estimators=10)
    >>> reg = RandomSeedAveragingRegressor(est)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y) # doctest: +ELLIPSIS
    RandomSeedAveragingRegressor(...)
    >>> y_pred = reg.predict(X)
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )

    def _check_params(self):
        super()._check_params()

        if not is_regressor(self.base_estimator):
            raise ValueError(f'base_estimator must be a regressor')

    def predict(self, X):
        self._check_is_fitted()

        predictions = np.asarray([e.predict(X) for e in self.estimators_]).T

        return np.average(predictions, axis=1)

class SplittedEstimator(BaseEstimator, MetaEstimatorMixin):
    @property
    def _estimator_type(self):
        return self.base_estimator._estimator_type

    @property
    def named_estimators_(self):
        self._check_is_fitted()

        return dict(zip(self.unique_groups_, self.estimators_))

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def _check_is_fitted(self):
        check_is_fitted(self, ['estimators_', 'unique_groups_'])

    def _check_params(self):
        if not is_estimator(self.base_estimator):
            raise ValueError(
                f'base_estimator must be a scikit-learn estimator'
            )

    def fit(self, X, y, by, **fit_params):
        X, y = check_X_y(X, y, estimator=self, force_all_finite='allow-nan')
        by = column_or_1d(by)

        check_consistent_length(X, by)

        self.unique_groups_ = np.unique(by)
        self.estimators_ = []

        for i in self.unique_groups_:
            e = clone(self.base_estimator)
            is_train = by == i

            e.fit(X[is_train], y[is_train], **fit_params)

            self.estimators_.append(e)

        return self

    def predict(self, X, by):
        self._check_is_fitted()

        X = check_array(X, estimator=self, force_all_finite='allow-nan')
        by = column_or_1d(by)

        check_consistent_length(X, by)

        y_pred = np.full_like(by, np.nan)

        for i, e in zip(self.unique_groups_, self.estimators_):
            is_test = by == i

            if np.sum(is_test) > 0:
                y_pred[is_test] = e.predict(X[is_test])

        return y_pred
