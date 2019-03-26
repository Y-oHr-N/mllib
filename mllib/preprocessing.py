from collections import Counter

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseEstimator


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
    def _kw_args(self):
        return {'a_min': self.data_min, 'a_max': self.data_max}

    def __init__(
        self,
        accept_sparse=False,
        data_max=np.inf,
        data_min=-np.inf,
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


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def _check_params(self):
        try:
            _ = np.dtype(self.dtype)
        except TypeError as e:
            raise ValueError(f'dtype must be a data type object') from e

    def _check_is_fitted(self):
        check_is_fitted(self, 'counters_')

    def fit(self, X, y=None):
        X = check_array(X, allow_nd=True, dtype=None, estimator=self)

        self.counters_ = [Counter(column) for column in X.T]

        return self

    def transform(self, X):
        self._check_is_fitted()

        X = check_array(X, allow_nd=True, dtype=None, estimator=self)
        Xt = np.empty_like(X, dtype=self.dtype)
        vectorized = np.vectorize(
            lambda counter, xj: counter[xj],
            excluded='counter'
        )

        for j, column in enumerate(X.T):
            Xt[:, j] = vectorized(self.counters_[j], column)

        return Xt
