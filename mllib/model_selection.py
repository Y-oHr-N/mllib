import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


class Objective:
    """Objective function.

    Parameters
    ----------
    estimator
        Object to use to fit the data.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    X
        Training data.

    y
        Target variable.

    cv
        Cross-validation strategy.

    error_score
        Value to assign to the score if an error occurs in estimator fitting.

    fit_params
        Parameters passed to the ``fit`` method of the estimator.

    return_train_score
        If True, training scores will be included.

    scoring
        String or callable to evaluate the predictions on the test data.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        X,
        y=None,
        cv=5,
        error_score='raise',
        fit_params=None,
        return_train_score=False,
        scoring=None
    ):
        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.param_distributions = param_distributions
        self.return_train_score = return_train_score
        self.scoring = scoring

    def __call__(self, trial):
        estimator = clone(self.estimator)
        params = {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }

        estimator.set_params(**params)

        cv_results = cross_validate(
            estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score=self.error_score,
            fit_params=self.fit_params,
            return_train_score=self.return_train_score,
            scoring=self.scoring
        )

        for k, v in cv_results.items():
            trial.set_user_attr(f'mean_{k}', np.mean(v))
            trial.set_user_attr(f'std_{k}', np.std(v))

        return - np.mean(cv_results['test_score'])


class TPESearchCV(BaseEstimator):
    """Hyper parameter search with cross-validation.

    Parameters
    ----------
    estimator
        Object to use to fit the data.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    cv
        Cross-validation strategy.

    error_score
        Value to assign to the score if an error occurs in estimator fitting.

    load_if_exists
        If True, the existing study is used in the case where a study named
        ``study_name`` already exists in the ``storage``.

    n_iter
        Number of trials.

    n_jobs
        Number of parallel jobs.

    random_state
        Seed of the pseudo random number generator.

    return_train_score
        If True, training scores will be included.

    refit
        If True, refit the estimator with the best found parameters.

    scoring
        String or callable to evaluate the predictions on the test data.

    storage
        Database URL.

    study_name
        name of the ``Study``.

    timeout
        Time limit in seconds for the search of appropriate models.

    verbose
        Verbosity level.

    Attributes
    ----------
    best_estimator_
        Estimator that was chosen by the search.

    scorer_
        Scorer function.

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
    """

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def best_params_(self):
        """Parameters of the best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self):
        """Mean cross-validated score of the best estimator.
        """

        self._check_is_fitted()

        return - self.best_value_

    @property
    def best_trial_(self):
        """Best trial in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def best_value_(self):
        """Best objective value in the ``Study``.
        """

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def decision_function(self):
        """Call decision_function on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        """Call inverse_transform on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        """Call predict on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        """Call predict_log_proba on the estimator with the best found
        parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        """Call predict_proba on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        """Call transform on the estimator with the best found parameters.
        """

        self._check_is_fitted()

        return self.best_estimator_.transform

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        error_score='raise',
        load_if_exists=False,
        n_iter=10,
        n_jobs=1,
        random_state=None,
        refit=True,
        return_train_score=False,
        scoring=None,
        storage=None,
        study_name=None,
        timeout=None,
        verbose=0
    ):
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self):
        check_is_fitted(self, ['best_estimator_', 'scorer_', 'study_'])

    def fit(self, X, y=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X
            Training data.

        y
            Target variable.

        **fit_params
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self
            Return self.
        """

        if self.verbose > 1:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
        elif self.verbose > 0:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        sampler = optuna.samplers.TPESampler(seed=seed)
        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=self.cv,
            error_score=self.error_score,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            scoring=self.scoring
        )

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = optuna.create_study(
            load_if_exists=self.load_if_exists,
            sampler=sampler,
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
            self.best_estimator_ = clone(self.estimator)

            self.best_estimator_.set_params(**self.study_.best_params)
            self.best_estimator_.fit(X, y)

        return self

    def score(self, X, y=None):
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

        self._check_is_fitted()

        return self.scorer_(self.best_estimator_, X, y)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
