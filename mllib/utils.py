from time import perf_counter

import numpy as np

optuna_is_installed = True
lgb_is_installed = True
xgb_is_installed = True

try:
    from optuna.distributions import (
        CategoricalDistribution,
        DiscreteUniformDistribution,
        IntUniformDistribution,
        LogUniformDistribution,
        UniformDistribution
    )
except ImportError:
    optuna_is_installed = False

try:
    import lightgbm as lgb
except ImportError:
    lgb_is_installed = False

try:
    import xgboost as xgb
except ImportError:
    xgb_is_installed = False

DEFAULT_N_TRIALS = 10


def check_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = np.asarray(sample_weight)

    if sample_weight.size != n_samples:
        raise ValueError(f'the size of sample_weight must be {n_samples}')

    if np.any(sample_weight < 0):
        raise ValueError(f'individual weights for each sample must be >= 0')

    return sample_weight


def compute_execution_time(func, *args, **kwargs):
    n_trials = kwargs.pop('n_trials', DEFAULT_N_TRIALS)
    start_time = perf_counter()

    for _ in range(n_trials):
        func(*args, **kwargs)

    return (perf_counter() - start_time) / n_trials


def get_param_distributions(estimator_name):
    """Get a dictionary where keys are parameters and values are
    distributions.

    Examples
    --------
    >>> from mllib.utils import get_param_distributions
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> param_distributions = get_param_distributions(clf.__class__.__name__)
    """

    if not optuna_is_installed:
        raise ImportError('optuna is not installed')

    dict_of_param_distributions = {
        'GradientBoostingClassifier': {
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'loss': CategoricalDistribution(['deviance', 'exponential']),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20),
            'subsample': UniformDistribution(0.5, 1.0)
        },
        'RandomForestClassifier': {
            'bootstrap': CategoricalDistribution([True, False]),
            'criterion': CategoricalDistribution(['entropy', 'gini']),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20)
        },
        'GradientBoostingRegressor': {
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'loss': CategoricalDistribution(
                ['huber', 'lad', 'ls', 'quantile']
            ),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20),
            'subsample': UniformDistribution(0.5, 1.0)
        },
        'RandomForestRegressor': {
            'bootstrap': CategoricalDistribution([True, False]),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20)
        }
    }

    if lgb_is_installed:
        dict_of_param_distributions['LGBMClassifier'] = {
            'boosting_type': CategoricalDistribution(['gbdt', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        }
        dict_of_param_distributions['LGBMRegressor'] = {
            'boosting_type': CategoricalDistribution(['gbdt', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        }

    if xgb_is_installed:
        dict_of_param_distributions['XGBClassifier'] = {
            'booster': CategoricalDistribution(['gbtree', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        }
        dict_of_param_distributions['XGBRegressor'] = {
            'booster': CategoricalDistribution(['gbtree', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        }

    return dict_of_param_distributions[estimator_name]


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
