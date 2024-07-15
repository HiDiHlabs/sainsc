from collections.abc import Callable

import anndata as ad
import pandas as pd


def celltype_signatures(
    adata: ad.AnnData,
    *,
    celltype_col: str = "leiden",
    agg_method: str | Callable = "mean",
) -> pd.DataFrame:
    """
    Calculate gene expression signatures per 'celltype'.

    Note, that this will make a dense copy of `adata.X` therefore potentially leading
    to large memory usage.

    Parameters
    ----------
    adata : anndata.AnnData
    celltype_col : str, optional
        Name of column in :py:attr:`anndata.AnnData.obs` containing cell-type
        information.
    agg_method : str or collections.abc.Callable, optional
        Function to aggregate gene expression per cluster used by
        :py:meth:`pandas.DataFrame.agg`.

    Returns
    -------
    pandas.DataFrame
        :py:class:`pandas.DataFrame` of gene expression aggregated per 'celltype'.
    """
    signatures = (
        adata.to_df()
        .merge(adata.obs[celltype_col], left_index=True, right_index=True)
        .groupby(celltype_col, observed=True, sort=False)
        .agg(agg_method)
        .transpose()
        .rename_axis(adata.var_names.name)
    )

    signatures /= signatures.sum(axis=0)

    return signatures
