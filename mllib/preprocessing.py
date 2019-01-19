import numpy as np
from sklearn.preprocessing import FunctionTransformer

__all__ = ['Clip', 'Affine', 'Log1P', 'Round', 'affine']


def affine(X, A=1.0, b=0.0, inverse=False):
    X = np.asarray(X)
    A = np.asarray(A)
    b = np.asarray(b)

    if inverse:
        if A.ndim < 2:
            return (X - b) / A
        else:
            return (X - b) @ np.linalg.inv(A)
    else:
        if A.ndim < 2:
            return X * A + b
        else:
            return X @ A + b


class Affine(FunctionTransformer):
    @property
    def _inv_kw_args(self):
        return {'A': self.A, 'b': self.b, 'inverse': True}

    @property
    def _kw_args(self):
        return {'A': self.A, 'b': self.b}

    def __init__(
        self,
        accept_sparse=False,
        A=1.0,
        b=0.0,
        validate=True
    ):
        self.A = A
        self.b = b

        super().__init__(
            accept_sparse=accept_sparse,
            check_inverse=False,
            func=affine,
            inverse_func=affine,
            inv_kw_args=self._inv_kw_args,
            kw_args=self._kw_args,
            validate=validate
        )

    def set_params(self, **params):
        super().set_params(**params)

        self.inv_kw_args = self._inv_kw_args
        self.kw_args = self._kw_args


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


if '__name__' == '__main__':
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(Affine)
    check_estimator(Clip)
    check_estimator(Log1P)
    check_estimator(Round)
