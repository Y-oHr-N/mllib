from time import perf_counter

DEFAULT_N_TRIALS = 10


def compute_execution_time(func, *args, **kwargs):
    n_trials = kwargs.pop('n_trials', DEFAULT_N_TRIALS)
    start_time = perf_counter()

    for _ in range(n_trials):
        func(*args, **kwargs)

    return (perf_counter() - start_time) / n_trials


def get_param_distributions(estimator_name):
    from optuna.distributions import (
        CategoricalDistribution,
        DiscreteUniformDistribution,
        IntUniformDistribution,
        LogUniformDistribution
    )

    dict_of_param_distributions = {
        'GradientBoostingClassifier': {
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'loss': CategoricalDistribution(['deviance', 'exponential']),
            'max_depth': IntUniformDistribution(3, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20),
            'subsample': DiscreteUniformDistribution(0.05, 1.0, 0.05)
        },
        'RandomForestClassifier': {
            'bootstrap': CategoricalDistribution([True, False]),
            'class_weight': CategoricalDistribution(['balanced', None]),
            'criterion': CategoricalDistribution(['entropy', 'gini']),
            'max_depth': IntUniformDistribution(3, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20)
        },
        'GradientBoostingRegressor': {
            'learning_rate': LogUniformDistribution(0.001, 1.0),
            'loss': CategoricalDistribution(
                ['huber', 'lad', 'ls', 'quantile']
            ),
            'max_depth': IntUniformDistribution(3, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20),
            'subsample': DiscreteUniformDistribution(0.05, 1.0, 0.05)
        },
        'RandomForestRegressor': {
            'bootstrap': CategoricalDistribution([True, False]),
            'max_depth': IntUniformDistribution(3, 10),
            'max_features': DiscreteUniformDistribution(0.05, 1.0, 0.05),
            'min_samples_leaf': IntUniformDistribution(1, 20),
            'min_samples_split': IntUniformDistribution(2, 20)
        }
    }

    return dict_of_param_distributions[estimator_name]


def is_lgbm_model(estimator):
    try:
        import lightgbm as lgb

        while hasattr(estimator, '_final_estimator'):
            estimator = estimator._final_estimator

        return isinstance(estimator, lgb.LGBMModel)

    except ImportError:
        return False


def is_xgb_model(estimator):
    try:
        import xgboost as xgb

        while hasattr(estimator, '_final_estimator'):
            estimator = estimator._final_estimator

        return isinstance(estimator, xgb.XGBModel)

    except ImportError:
        return False
