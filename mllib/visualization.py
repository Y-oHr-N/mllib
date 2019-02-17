import numpy as np
from sklearn.model_selection._validation import _translate_train_sizes
from sklearn.utils import check_X_y
from yellowbrick import reset_orig
from yellowbrick.base import ModelVisualizer

from .utils import compute_execution_time

reset_orig()

DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 5)


class TrainingTimeCurve(ModelVisualizer):
    def __init__(
        self,
        estimator,
        ax=None,
        n_trials=10,
        title='Training time curve',
        train_sizes=DEFAULT_TRAIN_SIZES
    ):
        super().__init__(estimator, ax=ax, title=title)

        self.n_trials = n_trials
        self.train_sizes = train_sizes

    def fit(self, X, y=None, **fit_params):
        X, y = check_X_y(X, y, estimator=self)
        n_samples, _ = X.shape

        self.train_sizes_abs_ = _translate_train_sizes(
            self.train_sizes,
            n_samples
        )
        self.fit_times_ = np.asarray(
            [
                compute_execution_time(
                    self.estimator.fit,
                    X[:n_train_samples],
                    y[:n_train_samples],
                    n_trials=self.n_trials,
                    **fit_params
                ) for n_train_samples in self.train_sizes_abs_
            ]
        )

        self.draw()

        return self

    def draw(self, **kwargs):
        kwargs['label'] = self.estimator.__class__.__name__
        kwargs['marker'] = 'o'

        self.ax.plot(self.train_sizes_abs_, self.fit_times_, **kwargs)

        return self.ax

    def finalize(self, **kwargs):
        xmin = 0.0
        xmax = 1.1 * np.max(self.train_sizes_abs_)
        ymin = 0.9 * np.min(self.fit_times_)
        ymax = 1.1 * np.max(self.fit_times_)

        kwargs['xlabel'] = 'Number of training samples'
        kwargs['xlim'] = (xmin, xmax)
        kwargs['ylabel'] = 'Time'
        kwargs['ylim'] = (ymin, ymax)
        kwargs['yscale'] = 'log'

        self.set_title()
        self.ax.grid(which='both')
        self.ax.legend(loc='upper left')
        self.ax.set(**kwargs)

        return self.ax
