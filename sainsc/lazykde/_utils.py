from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import csr_array, sparray, spmatrix
from skimage.measure import label, regionprops

from .._typealias import _AssignmentScoreMap, _CosineMap, _Kernel, _SignatureArray
from .._utils import _get_coordinate_index
from .._utils_rust import GridCounts

type _Shape = tuple[int, ...]

SCALEBAR_PARAMS = dict(frameon=False, color="w")
"""Default scalebar parameters"""


def _get_cell_dtype(n: int) -> np.dtype[np.signedinteger]:
    return np.result_type(np.int8, n)


def _filter_blobs[T: np.bool_ | np.integer](
    labeled_map: np.ndarray[_Shape, np.dtype[T]], min_blob_area: int
) -> np.ndarray[_Shape, np.dtype[T]]:
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
    coord: tuple[np.ndarray[tuple[int], np.dtype[np.integer]], ...],
    *,
    name: str | None = None,
    n_threads: int = 1,
) -> AnnData:
    obs = pd.DataFrame(
        index=_get_coordinate_index(*coord, name=name, n_threads=n_threads)
    )

    return AnnData(
        X=csr_array(kde),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genelist, name="gene")),
        obsm={"spatial": np.column_stack(coord)},
    )


class CosineCelltypeCallable(Protocol):
    def __call__(
        self,
        counts: GridCounts,
        genes: list[str],
        signatures: _SignatureArray,
        kernel: _Kernel,
        *,
        log: bool = ...,
        min_transcripts: int | None = ...,
        chunk_size: tuple[int, int] = ...,
        n_threads: int | None = ...,
    ) -> tuple[
        _CosineMap,
        _AssignmentScoreMap,
        np.ndarray[tuple[int, int], np.dtype[np.signedinteger]],
    ]: ...
