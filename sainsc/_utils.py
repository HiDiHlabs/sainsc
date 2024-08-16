import functools
import os
from typing import Callable, NoReturn, ParamSpec, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._utils_rust import coordinate_as_string


def _get_n_cpus() -> int:
    available_cpus = len(os.sched_getaffinity(0))
    return min(available_cpus, 32)


P = ParamSpec("P")
T = TypeVar("T")


def _validate_n_threads(n_threads: int | None) -> int:
    if n_threads is None:
        n_threads = 0
    if n_threads < 0:
        raise ValueError("`n_threads` must be >= 0.")
    else:
        return n_threads if n_threads > 0 else _get_n_cpus()


def validate_threads(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        n_threads = kwargs.get("n_threads", 0)
        assert n_threads is None or isinstance(n_threads, int)
        kwargs["n_threads"] = _validate_n_threads(n_threads)
        return func(*args, **kwargs)

    return wrapper


def _get_coordinate_index(
    x: NDArray[np.integer],
    y: NDArray[np.integer],
    *,
    name: str | None = None,
    n_threads: int | None = None,
) -> pd.Index:
    x_i32: NDArray[np.int32] = x.astype(np.int32, copy=False)
    y_i32: NDArray[np.int32] = y.astype(np.int32, copy=False)

    return pd.Index(
        coordinate_as_string(x_i32, y_i32, n_threads=n_threads), dtype=str, name=name
    )


def _raise_module_load_error(e: Exception, fn: str, pkg: str, extra: str) -> NoReturn:
    raise ModuleNotFoundError(
        f"`{fn}` requires '{pkg}' to be installed, e.g. via the '{extra}' extra."
    ) from e
