from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state


class BaseRandomSeedAveraging(BaseEstimator, ABC):
    # TODO: add a RandomSeedAveragingClassifier class
    # TODO: add a n_jobs parameter
    # TODO: add a verbose parameter

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        random_state=None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _check_params(self):
        params = self.base_estimator.get_params()

        if 'random_state' not in params:
            raise ValueError(
                f'base_estimator must have random_state'
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f'n_estimators must be > 0, got {self.n_estimators}'
            )

    def fit(self, X, y, **fit_params):
        self._check_params()

        random_state = check_random_state(self.random_state)

        self.estimators_ = []

        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            seed = random_state.randint(0, np.iinfo(np.int32).max)

            estimator.set_params(random_state=seed)
            estimator.fit(X, y, **fit_params)

            self.estimators_.append(estimator)

        return self

    @abstractmethod
    def predict(self, X):
        pass


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
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        random_state=None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def predict(self, X):
        return np.average([e.predict(X) for e in self.estimators_])


if __name__ == '__main__':
    import doctest

    doctest.testmod()
