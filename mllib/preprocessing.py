import numpy as np
from sklearn.preprocessing import FunctionTransformer

__all__ = ['Clip', 'LinearTransformer', 'Log1P', 'Round', 'linear_transform']


def linear_transform(X, add=0.0, multiply=1.0):
    return add + multiply * X


class Clip(FunctionTransformer):
    @property
    def _data_max(self):
        if self.data_max is None:
            return np.inf

        return self.data_max

    @property
    def _data_min(self):
        if self.data_min is None:
            return -np.inf

        return self.data_min

    @property
    def _kw_args(self):
        return {'a_min': self._data_min, 'a_max': self._data_max}

    def __init__(
        self,
        accept_sparse=False,
        data_max=None,
        data_min=None,
        validate=True
    ):
        self.data_max = data_max
        self.data_min = data_min

        super().__init__(
            accept_sparse=accept_sparse,
            check_inverse=False,
            func=np.clip,
            kw_args=self._kw_args,
            validate=validate
        )

    def set_params(self, **params):
        super().set_params(**params)

        self.kw_args = self._kw_args


class Round(FunctionTransformer):
    @property
    def _kw_args(self):
        return {'decimals': self.decimals}

    def __init__(
        self,
        accept_sparse=False,
        decimals=0,
        validate=True
    ):
        self.decimals = decimals

        super().__init__(
            accept_sparse=accept_sparse,
            check_inverse=False,
            func=np.round,
            kw_args=self._kw_args,
            validate=validate
        )

    def set_params(self, **params):
        super().set_params(**params)

        self.kw_args = self._kw_args


class LinearTransformer(FunctionTransformer):
    @property
    def _kw_args(self):
        return {'add': self.add, 'multiply': self.multiply}

    def __init__(
        self,
        accept_sparse=False,
        add=0.0,
        multiply=1.0,
        validate=True
    ):
        self.add = add
        self.multiply = multiply

        super().__init__(
            accept_sparse=accept_sparse,
            check_inverse=False,
            func=linear_transform,
            kw_args=self._kw_args,
            validate=validate
        )

    def set_params(self, **params):
        super().set_params(**params)

        self.kw_args = self._kw_args


class Log1P(FunctionTransformer):
    def __init__(
        self,
        accept_sparse=False,
        validate=True
    ):
        super().__init__(
            accept_sparse=accept_sparse,
            check_inverse=False,
            func=np.log1p,
            inverse_func=np.expm1,
            validate=validate
        )


if '__name__' == '__main__':
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(Clip)
    check_estimator(Round)
    check_estimator(LinearTransformer)
    check_estimator(Log1P)
