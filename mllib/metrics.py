import numpy as np
from sklearn.metrics import mean_squared_error

__all__ = ['root_mean_squared_error']


def root_mean_squared_error(y_true, y_pred, **kwargs):
    mse = mean_squared_error(y_true, y_pred, **kwargs)

    return np.sqrt(mse)
