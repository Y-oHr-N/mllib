from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator as SKLearnBaseEstimator

lgb_is_installed = True
xgb_is_installed = True

try:
    import lightgbm as lgb
except ImportError:
    lgb_is_installed = False

try:
    import xgboost as xgb
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
    def fit(self, X, y=None):
        pass
