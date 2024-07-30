from collections.abc import Iterable

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import csc_array, csr_array

from .. import GridCounts, GridFloats
from .._utils_rust import correct_ambient_rna_rs


def correct_ambient_rna(
    counts: GridCounts,
    raw: AnnData,
    corrected: AnnData,
    bin_ratio: int,
    *,
    n_threads: int | None = None,
) -> GridFloats:
    """
    Remove ambient RNA after running CellBender.

    The idea is to use :ref:`CellBender's <cellbender:introduction>` module
    ``remove-background`` on binned data to get estimates of the ambient RNA and then
    project it back to the pixels (or smaller bins) to remove the ambient RNA with
    increased resolution.

    The bin size of `raw` should be a multiple of `counts` otherwise pixels in `counts`
    can not be correctly mapped to the corresponding bins in `raw`/`corrected`.

    Parameters
    ----------
    counts : sainsc.GridCounts
        Raw counts (or small bins).
    raw : anndata.AnnData
        Raw bins of reasonable cell size before running CellBender.
    corrected : anndata.AnnData
        Ambient corrected bins (CellBender output).
    bin_ratio : int
        The ratio of the bin size to pixel size in `counts`, i.e. a `bin_ratio` of 20
        indicates that each bin is the size of 20 pixels.

    Returns
    -------
    sainsc.GridFloats
        Corrected RNA counts.
    """

    def scale_as_gridfloats(
        scaling: csc_array,
        genes: Iterable[str],
        x: NDArray[np.integer],
        y: NDArray[np.integer],
    ):
        shape = (x.max(), y.max())
        scale = {}
        for i, gene in enumerate(genes):
            gene_scale = scaling[:, i]
            scale[gene] = csr_array(
                (gene_scale.data, (x[gene_scale.indices], y[gene_scale.indices])), shape
            )
        return GridFloats(scale)

    if n_threads is None:
        n_threads = counts.n_threads

    # ensure ordering
    raw = raw[corrected.obs_names, corrected.var_names]

    # no element-wise division for sparse matrices -> multiply with X^-1
    scaling = csc_array(corrected.X.multiply(raw.X.astype(np.float32).power(-1)))
    scaling.eliminate_zeros()

    scale = scale_as_gridfloats(scaling, raw.var_names, *raw.obsm["spatial"].T)

    return correct_ambient_rna_rs(counts, scale, bin_ratio, n_threads=n_threads)
