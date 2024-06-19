import os
from typing import NoReturn

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sainsc._utils_rust import coordinate_as_string


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


def _bin_coordinates(df: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    df = df.assign(
        x=lambda df: _get_bin_coordinate(df["x"].to_numpy(), bin_size),
        y=lambda df: _get_bin_coordinate(df["y"].to_numpy(), bin_size),
    )
    return df


def _get_bin_coordinate(coor: NDArray[np.number], bin_size: float) -> NDArray[np.int32]:
    return np.floor(coor / bin_size).astype(np.int32, copy=False)


def _raise_module_load_error(e: Exception, fn: str, pkg: str, extra: str) -> NoReturn:
    raise ModuleNotFoundError(
        f"`{fn}` requires '{pkg}' to be installed, e.g. via the '{extra}' extra."
    ) from e
