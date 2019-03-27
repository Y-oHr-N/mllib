try:
    from lightgbm import LGBMModel

    LIGHTGBM_IS_INSTALLED = True

except ImportError as e:
    LGBMModel = object

    LIGHTGBM_IS_INSTALLED = False
    LIGHTGBM_IMPORT_ERROR = e

try:
    from optuna.distributions import CategoricalDistribution
    from optuna.distributions import DiscreteUniformDistribution
    from optuna.distributions import IntUniformDistribution
    from optuna.distributions import LogUniformDistribution
    from optuna.distributions import UniformDistribution

    OPTUNA_IS_INSTALLED = True

except ImportError as e:
    CategoricalDistribution = object
    DiscreteUniformDistribution = object
    IntUniformDistribution = object
    LogUniformDistribution = object
    UniformDistribution = object

    OPTUNA_IS_INSTALLED = False
    OPTUNA_IMPORT_ERROR = e

try:
    from xgboost import XGBModel

    XGBOOST_IS_INSTALLED = True

except ImportError as e:
    XGBModel = object

    XGBOOST_IS_INSTALLED = False
    XGBOOST_IMPORT_ERROR = e

try:
    from yellowbrick.base import ModelVisualizer
    from yellowbrick import reset_orig

    YELLOWBRICK_IS_INSTALLED = True

except ImportError as e:
    ModelVisualizer = object
    reset_orig = None

    YELLOWBRICK_IS_INSTALLED = False
    YELLOWBRICK_IMPORT_ERROR = e


def check_lightgbm_availability():
    if not LIGHTGBM_IS_INSTALLED:
        raise LIGHTGBM_IMPORT_ERROR


def check_optuna_availability():
    if not OPTUNA_IS_INSTALLED:
        raise OPTUNA_IMPORT_ERROR


def check_xgboost_availability():
    if not XGBOOST_IS_INSTALLED:
        raise XGBOOST_IMPORT_ERROR


def check_yellowbrick_availability():
    if not YELLOWBRICK_IS_INSTALLED:
        raise YELLOWBRICK_IMPORT_ERROR
