import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics.regression import _check_reg_targets

from .utils import check_sample_weight


def mean_arctangent_absolute_percentage_error(
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
    n_samples, _ = y_true.shape
    sample_weight = check_sample_weight(sample_weight, n_samples)

    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)

    is_nan = np.isnan(ape)
    ape[is_nan] = 0.0
    aape = np.arctan(ape)
    maape = np.average(aape, axis=0, weights=sample_weight)

    if multioutput == 'raw_values':
        return maape
    elif multioutput == 'uniform_average':
        multioutput = None

    return np.average(maape, weights=multioutput)


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput='uniform_average'
):
    """Mean Absolute Percentage Error (MAPE).

    Examples
    --------
    >>> from mllib.metrics import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    32.738...
    """

    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )

    n_samples, _ = y_true.shape
    sample_weight = check_sample_weight(sample_weight, n_samples)

    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)

    is_nan = np.isnan(ape)
    ape[is_nan] = 0.0
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
