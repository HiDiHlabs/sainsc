import gzip
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .._typealias import _PathLike
from .._utils_rust import categorical_coordinate


def _bin_coordinates(df: pl.DataFrame, bin_size: float) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col(i) - pl.col(i).min()).floordiv(bin_size).cast(pl.Int32, strict=True)
        for i in ["x", "y"]
    )
    return df


def _categorical_coordinate(
    x: NDArray[np.int32], y: NDArray[np.int32], *, n_threads: int | None = None
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    assert len(x) == len(y)

    return categorical_coordinate(x, y, n_threads=n_threads)


def _filter_genes(df: pl.DataFrame, remove_features: Collection[str]) -> pl.DataFrame:
    if len(remove_features) > 0:
        df = df.filter(
            ~pl.col("gene").cast(pl.Utf8).str.contains(f"({'|'.join(remove_features)})")
        )
    return df


# currently no need to support all file modes
_File_Mode = Literal["r", "w", "rb", "wb"]


def _open_file(file: _PathLike, mode: _File_Mode = "r"):
    file = Path(file)
    if file.suffix == ".gz":
        return gzip.open(file, mode)
    else:
        return open(file, mode)
