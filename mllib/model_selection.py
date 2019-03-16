import logging
from time import perf_counter
from typing import Any, Callable, Dict # noqa

import numpy as np
import pandas as pd # noqa
from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import BaseEstimator, is_estimator

optuna_is_installed = True

try:
    import optuna
except ImportError:
    optuna_is_installed = False


class Objective:
    """Callable that implements objective function.

    Parameters
    ----------
    estimator
        Object to use to fit the data. This is assumed to implement the
        scikit-learn estimator interface. Either this needs to provide
        ``score``, or ``scoring`` must be passed.

    param_distributions
        Dictionary where keys are parameters and values are distributions.
        Disributions are assumed to implement the optuna distribution
        interface.

    X
        Training data.

    y
        Target variable.

    cv
        Cross-validation strategy. Possible inputs for cv are:

        - integer to specify the number of folds in a CV splitter,
        - a CV splitter,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer, if ``estimator`` is a classifier and ``y`` is
        either binary or multiclass, ``StratifiedKFold`` is used. otherwise,
        ``KFold`` is used.

    error_score
        Value to assign to the score if an error occurs in fitting. If
        ‘raise’, the error is raised. If numeric, ``FitFailedWarning`` is
        raised. This does not affect the refit step, which will always raise
        the error.

    fit_params
        Parameters passed to ``fit`` one the estimator.

    groups
        Group labels for the samples used while splitting the dataset into
        train/test set.

    return_train_score
        If ``True``, training scores will be included. Computing training
        scores is used to get insights on how different hyperparameter
        settings impact the overfitting/underfitting trade-off. However
        computing training scores can be computationally expensive and is not
        strictly required to select the hyperparameters that yield the best
        generalization performance.

    scoring
        String or callable to evaluate the predictions on the test data. If
        ``None``, ``score`` on the estimator is used.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        X,
        y=None,
        cv=5,
        error_score=np.nan,
        fit_params=None,
        groups=None,
        return_train_score=False,
        scoring=None
    ):
        # type: (...) -> None

        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.groups = groups
        self.param_distributions = param_distributions
        self.return_train_score = return_train_score
        self.scoring = scoring

    def _cross_validate_with_pruning(self, trial, estimator):
        # type: (optuna.trial.Trial, BaseEstimator) -> Dict[str, np.ndarray]

        raise NotImplementedError(
            f'_cross_validate_with_pruning has not been implemented yet'
        )

    def _get_params(self, trial):
        # type: (optuna.trial.Trial) -> Dict[str, Any]

        return {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }

    def __call__(self, trial):
        # type: (optuna.trial.Trial) -> float

        estimator = clone(self.estimator)
        params = self._get_params(trial)

        estimator.set_params(**params)

        if hasattr(estimator, 'partial_fit'):
            cv_results = self._cross_validate_with_pruning(trial, estimator)
        else:
            cv_results = cross_validate(
                estimator,
                self.X,
                self.y,
                cv=self.cv,
                error_score=self.error_score,
                fit_params=self.fit_params,
                groups=self.groups,
                return_train_score=self.return_train_score,
                scoring=self.scoring
            )

        for name, array in cv_results.items():
            if name in ['test_score', 'train_score']:
                for i, score in enumerate(array):
                    trial.set_user_attr(f'split{i}_{name}', score)

            trial.set_user_attr(f'mean_{name}', np.average(array))
            trial.set_user_attr(f'std_{name}', np.std(array))

        return - trial.user_attrs['mean_test_score']


class TPESearchCV(BaseEstimator, MetaEstimatorMixin):
    """Hyperparameter search with cross-validation.

    Parameters
    ----------
    estimator
        Object to use to fit the data. This is assumed to implement the
        scikit-learn estimator interface. Either this needs to provide
        ``score``, or ``scoring`` must be passed.

    param_distributions
        Dictionary where keys are parameters and values are distributions.
        Disributions are assumed to implement the optuna distribution
        interface.

    cv
        Cross-validation strategy. Possible inputs for cv are:

        - integer to specify the number of folds in a CV splitter,
        - a CV splitter,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer, if ``estimator`` is a classifier and ``y`` is
        either binary or multiclass, ``StratifiedKFold`` is used. otherwise,
        ``KFold`` is used.

    error_score
        Value to assign to the score if an error occurs in fitting. If
        ‘raise’, the error is raised. If numeric, ``FitFailedWarning`` is
        raised. This does not affect the refit step, which will always raise
        the error.

    load_if_exists
        If ``True``, the existing study is used in the case where a study named
        ``study_name`` already exists in the ``storage``.

    n_iter
        Number of trials. If ``None``, there is no limitation on the number of
        trials. If ``timeout`` is also set to ``None``, the study continues to
        create trials until it receives a termination signal such as Ctrl+C or
        SIGTERM. This trades off runtime vs quality of the solution.

    n_jobs
        Number of parallel jobs. ``-1`` means using all processors.

    random_state
        Seed of the pseudo random number generator. If int, this is the seed
        used by the random number generator. If ``RandomState`` object, this
        is the random number generator. If ``None``, the random number
        generator is the ``RandomState`` object used by ``numpy.random``.

    return_train_score
        If ``True``, training scores will be included. Computing training
        scores is used to get insights on how different hyperparameter
        settings impact the overfitting/underfitting trade-off. However
        computing training scores can be computationally expensive and is not
        strictly required to select the hyperparameters that yield the best
        generalization performance.

    pruner
        Pruner that decides early stopping of unpromising trials. If ``None``,
        ``MedianPruner`` is used as the default.

    refit
        If ``True``, refit the estimator with the best found hyperparameters.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly.

    scoring
        String or callable to evaluate the predictions on the test data. If
        ``None``, ``score`` on the estimator is used.

    storage
        Database URL. If ``None``, in-memory storage is used, and the
        ``Study`` will not be persistent.

    study_name
        name of the ``Study``. If ``None``, a unique name is generated
        automatically.

    timeout
        Time limit in seconds for the search of appropriate models. If
        ``None``, the study is executed without time limitation. If
        ``n_trials`` is also set to ``None``, the study continues to create
        trials until it receives a termination signal such as Ctrl+C or
        SIGTERM. This trades off runtime vs quality of the solution.

    verbose
        Verbosity level. The higher, the more messages.

    Attributes
    ----------
    best_estimator_
        Estimator that was chosen by the search. This is present only if
        ``refit`` is set to ``True``.

    n_splits_
        Number of cross-validation splits.

    refit_time_
        Time for refitting the best estimator. This is present only if
        ``refit`` is set to ``True``.

    study_
        Study corresponds to the optimization task.

    Examples
    --------
    >>> from mllib.model_selection import TPESearchCV
    >>> from optuna.distributions import LogUniformDistribution
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> clf = SVC(gamma='auto')
    >>> param_distributions = {'C': LogUniformDistribution(1e-10, 1e+10)}
    >>> tpe_search = TPESearchCV(clf, param_distributions)
    >>> X, y = load_iris(return_X_y=True)
    >>> tpe_search.fit(X, y) # doctest: +ELLIPSIS
    TPESearchCV(...)
    >>> y_pred = tpe_search.predict(X)
    """

    @property
    def _estimator_type(self):
        # type: () -> str

        return self.estimator._estimator_type

    @property
    def _sampler(self):
        # type: () -> optuna.samplers.TPESampler

        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        return optuna.samplers.TPESampler(seed=seed)

    @property
    def best_index_(self):
        # type: () -> int
        """Index which corresponds to the best candidate parameter setting.
        """

        self._check_is_fitted()

        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_params_(self):
        # type: () -> Dict[str, Any]
        """Parameters of the best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self):
        # type: () -> float
        """Mean cross-validated score of the best estimator.
        """

        self._check_is_fitted()

        return - self.best_value_

    @property
    def best_trial_(self):
        # type: () -> optuna.structs.FrozenTrial
        """Best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def best_value_(self):
        # type: () -> float
        """Best objective value in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def classes_(self):
        # type: () -> np.ndarray
        """Class labels.
        """

        self._check_is_fitted()

        return self.best_estimator_.classes_

    @property
    def n_iter_(self):
        # type: () -> int
        """Actual number of trials.
        """

        self._check_is_fitted()

        return len(self.study_.trials)

    @property
    def scorer_(self):
        # type: () -> Callable[..., float]
        """Scorer function.
        """

        self._check_is_fitted()

        return check_scoring(self.estimator, scoring=self.scoring)

    @property
    def decision_function(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``decision_function`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``decision_function`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``inverse_transform`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``inverse_transform`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict`` on the estimator with the best found parameters.
        This is available only if the underlying estimator supports
        ``predict`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict_log_proba`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``predict_log_proba`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict_proba`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``predict_proba`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def score_samples(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``score_samples`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``score_samples`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.score_samples

    @property
    def transform(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``transform`` on the estimator with the best found parameters.
        This is available only if the underlying estimator supports
        ``transform`` and ``refit`` is set to ``True``.
        """

        self._check_is_fitted()

        return self.best_estimator_.transform

    @property
    def trials_dataframe(self):
        # type: () -> Callable[..., pd.DataFrame]
        """Call ``trials_dataframe`` on the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.trials_dataframe

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        error_score=np.nan,
        load_if_exists=False,
        n_iter=10,
        n_jobs=1,
        pruner=None,
        random_state=None,
        refit=True,
        return_train_score=False,
        scoring=None,
        storage=None,
        study_name=None,
        timeout=None,
        verbose=0
    ):
        # type: (...) -> None

        if not optuna_is_installed:
            raise ImportError('optuna is not installed')

        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.pruner = pruner
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self):
        # type: () -> None

        attributes = ['n_splits_', 'study_']

        if self.refit:
            attributes += ['best_estimator_', 'refit_time_']

        check_is_fitted(self, attributes)

    def _check_params(self):
        # type: () -> None

        if not is_estimator(self.estimator):
            raise ValueError(
                f'estimator must be a scikit-learn estimator'
            )

        if type(self.param_distributions) is not dict:
            raise ValueError(f'param_distributions must be a dictionary')

        for name, distribution in self.param_distributions.items():
            if not isinstance(
                distribution,
                optuna.distributions.BaseDistribution
            ):
                raise ValueError(
                    f'value of {name} must be a optuna distribution'
                )

    def _refit(self, X, y=None, **fit_params):
        # type: (np.ndarray, np.ndarray, Any) -> 'TPESearchCV'

        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.study_.best_params)

        start_time = perf_counter()

        self.best_estimator_.fit(X, y, **fit_params)

        self.refit_time_ = perf_counter() - start_time

        return self

    def _set_verbosity(self):
        # type: () -> None

        if self.verbose > 1:
            optuna.logging.set_verbosity(logging.DEBUG)
        elif self.verbose > 0:
            optuna.logging.set_verbosity(logging.INFO)
        else:
            optuna.logging.set_verbosity(logging.WARNING)

    def fit(self, X, y=None, groups=None, **fit_params):
        # type: (np.ndarray, np.ndarray, np.ndarray, Any) -> 'TPESearchCV'
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X
            Training data.

        y
            Target variable.

        groups
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params
            Parameters passed to ``fit`` on the estimator.

        Returns
        -------
        self
            Return self.
        """

        self._check_params()
        self._set_verbosity()

        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=self.cv,
            error_score=self.error_score,
            fit_params=fit_params,
            groups=groups,
            return_train_score=self.return_train_score,
            scoring=self.scoring
        )

        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)
        self.study_ = optuna.create_study(
            load_if_exists=self.load_if_exists,
            pruner=self.pruner,
            sampler=self._sampler,
            storage=self.storage,
            study_name=self.study_name
        )

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_iter,
            timeout=self.timeout
        )

        if self.refit:
            self._refit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        # type: (np.ndarray, np.ndarray) -> float
        """Return the score on the given data.

        Parameters
        ----------
        X
            Data.

        y
            Target variable.

        Returns
        -------
        score
            Scaler score.
        """

        return self.scorer_(self.best_estimator_, X, y)
