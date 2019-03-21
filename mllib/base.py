from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.externals.joblib import dump

try:
    import lightgbm as lgb

    lgb_is_installed = True

except ImportError:
    lgb_is_installed = False

try:
    import xgboost as xgb

    xgb_is_installed = True

except ImportError:
    xgb_is_installed = False


def is_estimator(estimator):
    return hasattr(estimator, 'fit')


def is_lgbm_model(estimator):
    if not lgb_is_installed:
        return False

    while hasattr(estimator, '_final_estimator'):
        estimator = estimator._final_estimator

    return isinstance(estimator, lgb.LGBMModel)


def is_xgb_model(estimator):
    if not xgb_is_installed:
        return False

    while hasattr(estimator, '_final_estimator'):
        estimator = estimator._final_estimator

    return isinstance(estimator, xgb.XGBModel)


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _check_params(self):
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        pass

    def to_pickle(self, filename, **kwargs):
        """Persist an estimator object.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path of the file in which it is to be stored.

        kwargs : dict
            Other keywords passed to ``sklearn.externals.joblib.dump``.

        Returns
        -------
        filenames : list
            List of file names in which the data is stored.
        """

        self._check_is_fitted()

        return dump(self, filename, **kwargs)
