import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics.regression import _check_reg_targets
from sklearn.utils.validation import check_consistent_length


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput='uniform_average'
):
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )

    check_consistent_length(y_true, y_pred, sample_weight)

    ape = 100.0 * np.abs((y_true - y_pred) / y_true)
    mape = np.average(ape, axis=0, weights=sample_weight)

    if multioutput == 'raw_values':
        return mape
    elif multioutput == 'uniform_average':
        multioutput = None

    return np.average(mape, weights=multioutput)


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
