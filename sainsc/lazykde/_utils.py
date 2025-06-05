from collections.abc import Iterable
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, sparray, spmatrix
from skimage.measure import label, regionprops

from .._utils import _get_coordinate_index
from .._utils_rust import GridCounts

T = TypeVar("T", bound=np.number)
U = TypeVar("U", bound=np.bool_ | np.integer)

SCALEBAR_PARAMS = dict(box_alpha=0, color="w")
"""Default scalebar parameters"""


def _get_cell_dtype(n: int) -> np.dtype[np.signedinteger]:
    return np.result_type(np.int8, n)


def _filter_blobs(labeled_map: NDArray[U], min_blob_area: int) -> NDArray[U]:
    # remove small blops (i.e. "cells")
    if min_blob_area <= 0:
        raise ValueError("Area must be bigger than 0")
    blob_labels = label(labeled_map, background=0)
    for blop in regionprops(blob_labels):
        if blop.area_filled < min_blob_area:
            min_x, min_y, max_x, max_y = blop.bbox
            labeled_map[min_x:max_x, min_y:max_y][blop.image] = 0
    return labeled_map


def _localmax_anndata(
    kde: spmatrix | sparray | NDArray,
    genelist: Iterable[str],
    coord: tuple[NDArray[np.integer], ...],
    *,
    name: str | None = None,
    n_threads: int = 1,
) -> AnnData:
    obs = pd.DataFrame(
        index=_get_coordinate_index(*coord, name=name, n_threads=n_threads)
    )

    return AnnData(
        X=csr_matrix(kde),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genelist, name="gene")),
        obsm={"spatial": np.column_stack(coord)},
    )


class CosineCelltypeCallable(Protocol):
    def __call__(
        self,
        counts: GridCounts,
        genes: list[str],
        signatures: NDArray[np.float32],
        kernel: NDArray[np.float32],
        *,
        log: bool = ...,
        chunk_size: tuple[int, int] = ...,
        n_threads: int | None = ...,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.signedinteger]]: ...
