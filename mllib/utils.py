from time import perf_counter

import numpy as np
from sklearn.utils import safe_indexing as sklearn_safe_indexing

try:
    from optuna.distributions import CategoricalDistribution
    from optuna.distributions import DiscreteUniformDistribution
    from optuna.distributions import IntUniformDistribution
    from optuna.distributions import LogUniformDistribution
    from optuna.distributions import UniformDistribution

    optuna_is_installed = True

except ImportError:
    optuna_is_installed = False

DEFAULT_N_TRIALS = 10


def add_prefix(dictionary, prefix=None):
    if prefix is None:
        return dictionary

    return {prefix + key: value for key, value in dictionary.items()}


def check_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = np.asarray(sample_weight)

    if sample_weight.ndim != 1:
        raise ValueError(f'sample_weight must be a 1D array')

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


def get_param_distributions(estimator_name, prefix=None):
    """Get distributions of parameters.

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
        'KBinsDiscretizer': {
            'n_bins': IntUniformDistribution(2, 10)
        },
        'GradientBoostingClassifier': {
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'loss': CategoricalDistribution(['deviance', 'exponential']),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20),
            'subsample': UniformDistribution(0.5, 1.0)
        },
        'KNeighborsClassifier': {
            'metric': CategoricalDistribution(['euclidean', 'manhattan']),
            'n_neighbors': IntUniformDistribution(1, 100),
            'weights': CategoricalDistribution(['distance', 'uniform'])
        },
        'LGBMClassifier': {
            'boosting_type': CategoricalDistribution(['gbdt', 'goss', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
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
        'XGBClassifier': {
            'booster': CategoricalDistribution(['gbtree', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
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
        'KNeighborsRegressor': {
            'metric': CategoricalDistribution(['euclidean', 'manhattan']),
            'n_neighbors': IntUniformDistribution(1, 100),
            'weights': CategoricalDistribution(['distance', 'uniform'])
        },
        'LGBMRegressor': {
            'boosting_type': CategoricalDistribution(['gbdt', 'goss', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        },
        'RandomForestRegressor': {
            'bootstrap': CategoricalDistribution([True, False]),
            'max_depth': IntUniformDistribution(1, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20)
        },
        'XGBRegressor': {
            'booster': CategoricalDistribution(['gbtree', 'dart']),
            'colsample_bytree': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'max_depth': IntUniformDistribution(1, 10),
            'reg_alpha': LogUniformDistribution(1e-06, 1.0),
            'reg_lambda': LogUniformDistribution(1e-6, 1.0),
            'subsample': UniformDistribution(0.5, 1.0)
        }
    }

    param_distributions = dict_of_param_distributions[estimator_name]

    return add_prefix(param_distributions, prefix)


def safe_indexing(X, indices):
    if X is None:
        return X
    else:
        return sklearn_safe_indexing(X, indices)
