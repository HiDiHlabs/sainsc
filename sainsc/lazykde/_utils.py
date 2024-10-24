from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from numba import njit
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, sparray, spmatrix
from skimage.measure import label, regionprops

from .._typealias import _Local_Max, _PathLike
from .._utils import _get_coordinate_index
from .._utils_rust import GridCounts

T = TypeVar("T", bound=np.number)
U = TypeVar("U", bound=np.bool_ | np.integer)

SCALEBAR_PARAMS = dict(box_alpha=0, color="w")
"""Default scalebar parameters"""


@njit
def _apply_color(
    img_in: NDArray[np.integer], cmap: tuple[NDArray[T], ...]
) -> NDArray[T]:
    img = np.empty(shape=(*img_in.shape, 3), dtype=cmap[0].dtype)
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            img[i, j, :] = cmap[img_in[i, j]]
    return img


def _get_cell_dtype(n: int) -> np.dtype:
    return np.result_type("int8", n)


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


def _load_localmax_cosine(
    coord: _Local_Max, zarr_store: _PathLike, *, celltypes: Iterable[str] | None = None
) -> pd.DataFrame:
    cosine_group = zarr.open_group(store=zarr_store, mode="r", path="cosine")

    celltypes = cosine_group.array_keys() if celltypes is None else celltypes

    cosine_df = pd.DataFrame({"x": coord[0], "y": coord[1]})

    for ct in celltypes:
        cosine_ct = cosine_group[ct]
        assert isinstance(cosine_ct, zarr.Array)
        cosine_df[ct] = cosine_ct.get_coordinate_selection(coord)

    return cosine_df


class CosineCelltypeCallable(Protocol):
    def __call__(
        self,
        counts: GridCounts,
        genes: list[str],
        celltypes: list[str],
        signatures: NDArray[np.float32],
        kernel: NDArray[np.float32],
        *,
        log: bool = ...,
        zarr_path: Path | None = None,
        chunk_size: tuple[int, int] = ...,
        n_threads: int | None = ...,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.signedinteger]]: ...
