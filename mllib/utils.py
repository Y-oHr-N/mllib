from time import perf_counter

from sklearn.pipeline import Pipeline

DEFAULT_N_TRIALS = 10


def is_lgbm_model(estimator):
    try:
        import lightgbm as lgb

        if isinstance(estimator, Pipeline):
            estimator = estimator._final_estimator

        return isinstance(estimator, lgb.LGBMModel)

    except ImportError:
        return False


def is_xgb_model(estimator):
    try:
        import xgboost as xgb

        if isinstance(estimator, Pipeline):
            estimator = estimator._final_estimator

        return isinstance(estimator, xgb.XGBModel)

    except ImportError:
        return False


def compute_execution_time(func, *args, **kwargs):
    n_trials = kwargs.pop('n_trials', DEFAULT_N_TRIALS)
    start_time = perf_counter()

    for _ in range(n_trials):
        func(*args, **kwargs)

    return (perf_counter() - start_time) / n_trials
