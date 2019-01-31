import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics.regression import _check_reg_targets


def root_mean_squared_error(
    y_true,
    y_pred,
    multioutput='uniform_average',
    sample_weight=None
):
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )
    mse = mean_squared_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput='raw_values'
    )
    rmse = np.sqrt(mse)

    if multioutput == 'raw_values':
        return rmse
    elif multioutput == 'uniform_average':
        multioutput = None

    return np.average(rmse, weights=multioutput)


def root_mean_squared_log_error(
    y_true,
    y_pred,
    multioutput='uniform_average',
    sample_weight=None
):
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )
    msle = mean_squared_log_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput='raw_values'
    )
    rmsle = np.sqrt(msle)

    if multioutput == 'raw_values':
        return rmsle
    elif multioutput == 'uniform_average':
        multioutput = None

    return np.average(rmsle, weights=multioutput)
