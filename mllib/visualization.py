import numpy as np
from sklearn.model_selection._validation import _translate_train_sizes

from .compat import ModelVisualizer
from .compat import reset_orig
from .compat import YELLOWBRICK_IS_INSTALLED
from .utils import compute_execution_time
from .utils import safe_indexing

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
        if not YELLOWBRICK_IS_INSTALLED:
            raise ImportError('yellowbrick is not installed')

        super().__init__(estimator, ax=ax, title=title)

        self.n_trials = n_trials
        self.train_sizes = train_sizes

    def fit(self, X, y=None, **fit_params):
        n_samples = len(X)

        self.train_sizes_abs_ = _translate_train_sizes(
            self.train_sizes,
            n_samples
        )
        self.fit_times_ = np.asarray([
            compute_execution_time(
                self.estimator.fit,
                safe_indexing(X, slice(0, n_train_samples)),
                safe_indexing(y, slice(0, n_train_samples)),
                n_trials=self.n_trials,
                **fit_params
            ) for n_train_samples in self.train_sizes_abs_
        ])

        self.draw()

        return self

    def draw(self, **kwargs):
        reset_orig()

        kwargs.setdefault('label', self.estimator.__class__.__name__)
        kwargs.setdefault('marker', 'o')

        self.ax.plot(self.train_sizes_abs_, self.fit_times_, **kwargs)

        return self.ax

    def finalize(self, **kwargs):
        xmin = 0.0
        xmax = 1.1 * np.max(self.train_sizes_abs_)
        ymin = 0.9 * np.min(self.fit_times_)
        ymax = 1.1 * np.max(self.fit_times_)

        kwargs.setdefault('xlabel', 'Number of training samples')
        kwargs.setdefault('xlim', (xmin, xmax))
        kwargs.setdefault('ylabel', 'Time')
        kwargs.setdefault('ylim', (ymin, ymax))
        kwargs.setdefault('yscale', 'log')

        self.set_title()
        self.ax.grid(which='both')
        self.ax.legend(loc='upper left')
        self.ax.set(**kwargs)

        return self.ax
