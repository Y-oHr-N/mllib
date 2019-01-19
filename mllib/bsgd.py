from abc import abstractmethod, ABC
from logging import getLogger
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.sgd_fast import (
    EpsilonInsensitive,
    Huber,
    LossFunction,
    SquaredLoss
)
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import (
    check_X_y,
    check_array,
    check_random_state,
    gen_batches,
    resample
)
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm, trange

__all__ = ['bsgd', 'BSGDRegressor']

LOGGER = getLogger(__name__)
LOSS_CLASSES = {
    'epsilon_insensitive': EpsilonInsensitive,
    'huber': Huber,
    'squared_loss': SquaredLoss
}


def bsgd(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: np.random.RandomState,
    support_vectors: np.ndarray,
    dual_coef: np.ndarray,
    intercept: float,
    t: int,
    alpha: float = 1e-04,
    coef0: float = 0.0,
    degree: int = 3,
    epsilon: float = 0.1,
    eta0: float = 0.01,
    fit_intercept: bool = True,
    gamma: float = None,
    kernel: str = 'linear',
    learning_rate: str = 'invscaling',
    loss: str = 'squared_loss',
    max_iter: int = 5,
    max_support: int = 100,
    power_t: float = 0.25,
    shuffle: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Budgeted Stochastic Gradient Descent (BSGD).

    Parameters
    ----------
    X
        Training Data.

    y
        Target values.

    sample_weight
        Weights applied to individual samples.

    random_state
        Seed of the pseudo random number generator.

    support_vectors
        Initial support vectors.

    dual_coef
        Initial coefficients of support vectors.

    intercept
        Initial intercept term.

    t
        Initial round.

    alpha
        Regularization parameter.

    coef0
        Independent term in the polynomial (or sigmoid) kernel function.

    degree
        Degree of the polynomial kernel function.

    epsilon
        Epsilon in the epsilon-insensitive (or Huber) loss functions.

    eta0
        Initial learning rate.

    fit_intercept
        If True, fit the intercept.

    gamma
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    kernel
        Used kernel function.

    learning_rate
        Learning rate schedule.

    loss
        Used Loss function.

    max_iter
        Maximum number of epochs.

    max_support
        Maximum number of support vectors (a.k.a. budget).

    power_t
        Exponent for inverse scaling learning rate.

    shuffle
        If True, shuffle the data.

    verbose
        Enable verbose output.

    Returns
    -------
    support_vectors
        Support vectors.

    dual_coef
        Coefficients of support vectors.

    intercept
        Intercept term.

    t
        Round.
    """

    loss_function = _get_loss_function(loss=loss, epsilon=epsilon)
    disable = not(verbose)
    epochs = trange(max_iter, disable=disable)
    n_samples, _ = X.shape
    generator = gen_batches(n_samples, 1)
    batches = tqdm(generator, disable=disable, leave=False)

    for _ in epochs:
        if shuffle:
            X, y, sample_weight = resample(
                X,
                y,
                sample_weight,
                random_state=random_state,
                replace=False
            )

        for batch in batches:
            X_batch = X[batch]
            y_batch = y[batch]
            sample_weight_batch = sample_weight[batch]
            y_score = _decision_function(
                X_batch,
                support_vectors,
                dual_coef,
                intercept,
                coef0=coef0,
                degree=degree,
                gamma=gamma,
                kernel=kernel
            )

            eta = _get_eta(
                t,
                eta0=eta0,
                learning_rate=learning_rate,
                power_t=power_t
            )

            dual_coef -= eta * alpha * dual_coef

            dloss = loss_function.dloss(y_score, y_batch)
            update = - sample_weight_batch * eta * dloss

            if update != 0.0:
                condition = np.all(support_vectors == X_batch, axis=1)
                indices = np.flatnonzero(condition)

                if indices.size > 0:
                    index = indices[0]
                    dual_coef[index] += update
                else:
                    support_vectors = np.append(
                        support_vectors,
                        X_batch,
                        axis=0
                    )
                    dual_coef = np.append(dual_coef, update)

                if fit_intercept:
                    intercept += update

                LOGGER.info('added the support vector')

            n_SV, _ = support_vectors.shape

            if n_SV > max_support:
                abs_dual_coef = np.abs(dual_coef)
                removed = np.argmin(abs_dual_coef)
                support_vectors = np.delete(support_vectors, removed, axis=0)
                dual_coef = np.delete(dual_coef, removed)

                LOGGER.info(f'removed the {removed + 1}-th support vector')

            t += 1

    return support_vectors, dual_coef, intercept, t


def _get_loss_function(
    loss: str = 'squared_loss',
    epsilon: float = 0.1
) -> LossFunction:

    loss_class = LOSS_CLASSES[loss]

    if loss in ['epsilon_insensitive', 'huber']:
        return loss_class(epsilon)

    return loss_class()


def _get_eta(
    t: int,
    eta0: float = 0.01,
    learning_rate: str = 'invscaling',
    power_t: float = 0.25
) -> float:

    if learning_rate == 'constant':
        return eta0

    return eta0 / t ** power_t


def _decision_function(
    X: np.ndarray,
    support_vectors: np.ndarray,
    dual_coef: np.ndarray,
    intercept: float,
    coef0: float = 0.0,
    degree: int = 3,
    gamma: float = None,
    kernel: str = 'linear'
) -> np.ndarray:

    n_SV, _ = support_vectors.shape

    if n_SV == 0:
        n_samples, _ = X.shape

        return np.zeros(n_samples)

    K = pairwise_kernels(
        X,
        support_vectors,
        coef0=coef0,
        degree=degree,
        filter_params=True,
        gamma=gamma,
        metric=kernel
    )

    return K @ dual_coef + intercept


class BSGD(BaseEstimator, ABC):
    # TODO: add a BNormaClassifier class
    # TODO: add a BNormaRegressor class
    # TODO: add a BPegasosClassifier class
    # TODO: add a BPegasosRegressor class
    # TODO: add a sumloss_ attribute
    # TODO: add a batch_size parameter
    # TODO: add a n_iter_no_change parameter
    # TODO: add a n_jobs parameter
    # TODO: add a strategy parameter
    # TODO: add a tol parameter
    # TODO: correct the shape of a dual_coef_ attribute
    # TODO: correct the shape of a intercept_ attribute
    # TODO: implement a bsgd function with Cython (or Numba)
    # TODO: implement a nbsgd function
    # TODO: write doctest examples

    @abstractmethod
    def __init__(
        self,
        loss: str,
        alpha: float = 1e-04,
        coef0: float = 0.0,
        degree: int = 3,
        epsilon: float = 0.1,
        eta0: float = 0.01,
        fit_intercept: bool = True,
        gamma: float = None,
        kernel: str = 'linear',
        learning_rate: str = 'invscaling',
        max_iter: int = 5,
        max_support: int = 100,
        power_t: float = 0.25,
        random_state: Union[int, np.random.RandomState] = None,
        shuffle: bool = True,
        verbose: bool = False,
        warm_start: bool = False
    ) -> None:

        self.alpha = alpha
        self.coef0 = coef0
        self.degree = degree
        self.epsilon = epsilon
        self.eta0 = eta0
        self.fit_intercept = fit_intercept
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.loss = loss
        self.kernel = kernel
        self.max_iter = max_iter
        self.max_support = max_support
        self.power_t = power_t
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.warm_start = warm_start

    def _check_params(self) -> None:
        if self.alpha < 0.0:
            raise ValueError(f'alpha must be >= 0, got {self.alpha}')

        if self.eta0 <= 0.0:
            raise ValueError(f'eta0 must be > 0, got {self.eta0}')

        if self.eta0 * self.alpha >= 1.0:
            raise ValueError(f'eta0 must be < 1 / alpha, got {self.eta0}')

        if not isinstance(self.fit_intercept, bool):
            raise ValueError(f'fit_intercept must be either True or False')

        if self.learning_rate not in ('constant', 'invscaling'):
            raise ValueError(
                f'learning rate {self.learning_rate} is not supported'
            )

        if self.loss not in self._losses:
            raise ValueError(f'loss {self.loss} is not supported')

        if self.max_iter <= 0:
            raise ValueError(f'max_iter must be > 0, got {self.max_iter}')

        if self.max_support <= 0:
            raise ValueError(
                f'max_support must be > 0, got {self.max_support}'
            )

        if self.power_t < 0.0:
            raise ValueError(f'power_t must be >= 0, got {self.power_t}')

        if not isinstance(self.shuffle, bool):
            raise ValueError(f'shuffle must be either True or False')

        if not isinstance(self.verbose, bool):
            raise ValueError(f'verbose must be either True or False')

        if not isinstance(self.warm_start, bool):
            raise ValueError(f'warm_start must be either True or False')

    def _check_sample_weight(
        self,
        sample_weight: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)

        if sample_weight.size != n_samples:
            raise ValueError('shapes of X and sample_weight do not match')

        return sample_weight

    def _reset(self) -> None:
        if not self.warm_start and hasattr(self, 'support_vectors_'):
            del self.support_vectors_
            del self.dual_coef_
            del self.intercept_

        self.t_ = 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> BaseEstimator:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target values.

        sample_weight
            Weights applied to individual samples.

        Returns
        -------
        self
            Return self.
        """

        self._reset()

        return self._partial_fit(X, y, sample_weight, self.max_iter)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> BaseEstimator:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target values.

        sample_weight
            Weights applied to individual samples.

        Returns
        -------
        self
            Return self.
        """

        return self._partial_fit(X, y, sample_weight, 1)

    @abstractmethod
    def _partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        max_iter: int
    ) -> BaseEstimator:

        pass


class BSGDRegressor(BSGD, RegressorMixin):
    """BSGD Regressor.

    Parameters
    ----------
    alpha
        Regularization parameter.

    coef0
        Independent term in the polynomial (or sigmoid) kernel function.

    degree
        Degree of the polynomial kernel function.

    epsilon
        Epsilon in the epsilon-insensitive (or Huber) loss functions.

    eta0
        Initial learning rate.

    fit_intercept
        If True, fit the intercept.

    gamma
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    kernel
        Used kernel function.

    learning_rate
        Learning rate schedule.

    loss
        Used Loss function.

    max_iter
        Maximum number of epochs.

    max_support
        Maximum number of support vectors (a.k.a. budget).

    power_t
        Exponent for inverse scaling learning rate.

    random_state
        Seed of the pseudo random number generator.

    shuffle
        If True, shuffle the data.

    verbose
        Enable verbose output.

    warm_start
        If True, reuse the solution of the previous call to fit as
        initialization.

    Attributes
    ----------
    support_vectors_
        Support vectors.

    dual_coef_
        Coefficients of support vectors.

    intercept_
        Intercept term.

    t_
        Round.

    References
    ----------
    Wang, Z., Crammer, K. and Vucetic, S.
    Breaking the curse of kernelization: Budgeted stochastic gradient descent
    for large-scale svm training.
    Journal of Machine Learning Research (JMLR), Vol. 13, pp. 3103-3131, 2012.
    """

    _losses = ['epsilon_insensitive', 'huber', 'squared_loss']

    def __init__(
        self,
        alpha: float = 1e-04,
        coef0: float = 0.0,
        degree: int = 3,
        epsilon: float = 0.1,
        eta0: float = 0.01,
        fit_intercept: bool = True,
        gamma: float = None,
        kernel: str = 'linear',
        learning_rate: str = 'invscaling',
        loss: str = 'squared_loss',
        max_iter: int = 5,
        max_support: int = 100,
        power_t: float = 0.25,
        random_state: Union[int, np.random.RandomState] = None,
        shuffle: bool = True,
        verbose: bool = False,
        warm_start: bool = False
    ) -> None:

        super().__init__(
            alpha=alpha,
            coef0=coef0,
            degree=degree,
            epsilon=epsilon,
            eta0=eta0,
            fit_intercept=fit_intercept,
            gamma=gamma,
            kernel=kernel,
            learning_rate=learning_rate,
            loss=loss,
            max_iter=max_iter,
            max_support=max_support,
            power_t=power_t,
            random_state=random_state,
            shuffle=shuffle,
            verbose=verbose,
            warm_start=warm_start
        )

    def _partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        max_iter: int
    ) -> BaseEstimator:

        self._check_params()

        X, y = check_X_y(X, y, estimator=self)
        n_samples, n_features = X.shape
        sample_weight = self._check_sample_weight(sample_weight, n_samples)
        random_state = check_random_state(self.random_state)

        if not hasattr(self, 'support_vectors_'):
            self.support_vectors_ = np.empty((0, n_features))
            self.dual_coef_ = np.empty(0)
            self.intercept_ = 0.0

        if not hasattr(self, 't_'):
            self.t_ = 1

        self.support_vectors_, self.dual_coef_, self.intercept_, self.t_ = \
            bsgd(
                X,
                y,
                sample_weight,
                random_state,
                self.support_vectors_,
                self.dual_coef_,
                self.intercept_,
                self.t_,
                alpha=self.alpha,
                coef0=self.coef0,
                degree=self.degree,
                epsilon=self.epsilon,
                eta0=self.eta0,
                fit_intercept=self.fit_intercept,
                gamma=self.gamma,
                kernel=self.kernel,
                learning_rate=self.learning_rate,
                loss=self.loss,
                max_iter=max_iter,
                max_support=self.max_support,
                power_t=self.power_t,
                shuffle=self.shuffle,
                verbose=self.verbose
            )

        return self

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """

        check_is_fitted(self, ['dual_coef_', 'support_vectors_', 't_'])

        X = check_array(X, estimator=self)

        return _decision_function(
            X,
            self.support_vectors_,
            self.dual_coef_,
            self.intercept_,
            coef0=self.coef0,
            degree=self.degree,
            gamma=self.gamma,
            kernel=self.kernel
        )


if '__name__' == '__main__':
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(BSGDRegressor)
