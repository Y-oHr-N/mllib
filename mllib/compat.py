try:
    from lightgbm import LGBMModel

    LGB_IS_INSTALLED = True

except ImportError:
    LGBMModel = object

    LGB_IS_INSTALLED = False

try:
    from optuna.distributions import CategoricalDistribution
    from optuna.distributions import DiscreteUniformDistribution
    from optuna.distributions import IntUniformDistribution
    from optuna.distributions import LogUniformDistribution
    from optuna.distributions import UniformDistribution

    OPTUNA_IS_INSTALLED = True

except ImportError:
    CategoricalDistribution = object
    DiscreteUniformDistribution = object
    IntUniformDistribution = object
    LogUniformDistribution = object
    UniformDistribution = object

    OPTUNA_IS_INSTALLED = False

try:
    from xgboost import XGBModel

    XGB_IS_INSTALLED = True

except ImportError:
    XGBModel = object

    XGB_IS_INSTALLED = False

try:
    from yellowbrick.base import ModelVisualizer
    from yellowbrick import reset_orig

    YELLOWBRICK_IS_INSTALLED = True

except ImportError:
    ModelVisualizer = object
    reset_orig = None

    YELLOWBRICK_IS_INSTALLED = False
