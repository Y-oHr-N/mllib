from time import perf_counter

__all__ = ['compute_execution_time']

DEFAULT_N_TRIALS = 10


def compute_execution_time(func, *args, **kwargs):
    n_trials = kwargs.pop('n_trials', DEFAULT_N_TRIALS)
    start_time = perf_counter()

    for _ in range(n_trials):
        func(*args, **kwargs)

    return (perf_counter() - start_time) / n_trials
