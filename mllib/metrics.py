import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def root_mean_squared_error(y_true, y_pred, **kwargs):
    mse = mean_squared_error(y_true, y_pred, **kwargs)

    return np.sqrt(mse)


def root_mean_squared_log_error(y_true, y_pred, **kwargs):
    msle = mean_squared_log_error(y_true, y_pred, **kwargs)

    return np.sqrt(msle)
