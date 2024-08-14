import os
from typing import NoReturn

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._utils_rust import coordinate_as_string


def _get_n_cpus() -> int:
    return len(os.sched_getaffinity(0))


def _get_coordinate_index(
    x: NDArray[np.integer],
    y: NDArray[np.integer],
    *,
    name: str | None = None,
    n_threads: int = 1,
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
