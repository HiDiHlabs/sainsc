from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike


def celltype_signatures(
    adata: ad.AnnData,
    *,
    celltype_col: str = "leiden",
    layer: str | None = None,
    dtype: DTypeLike = np.float32,
) -> pd.DataFrame:
    """
    Calculate gene expression signatures per 'cell type'.

    Parameters
    ----------
    adata : anndata.AnnData
    celltype_col : str, optional
        Name of column in :py:attr:`anndata.AnnData.obs` containing cell-type
        information.
    layer : str, optional
        Which :py:attr:`anndata.AnnData.layers` to use for aggregation. If `None`,
        :py:attr:`anndata.AnnData.X` is used.
    dytpe : numpy.typing.DTypeLike
        Data type to use for the signatures.

    Returns
    -------
    pandas.DataFrame
        :py:class:`pandas.DataFrame` of gene expression aggregated per 'cell type'.
    """
    X = adata.X if layer is None else adata.layers[layer]
    grouping = adata.obs.groupby(celltype_col, observed=True, sort=False).indices

    signatures: dict[Any, np.ndarray] = {}
    for name, indices in grouping.items():
        mean_X_group = X[indices].mean(axis=0, dtype=dtype)
        signatures[name] = (
            mean_X_group.A1 if isinstance(mean_X_group, np.matrix) else mean_X_group
        )

    signatures_df = pd.DataFrame(signatures, index=adata.var_names)
    signatures_df /= signatures_df.sum(axis=0)

    return signatures_df
