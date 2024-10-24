from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy.sparse import csr_matrix

from .._typealias import _PathLike
from .._utils import _get_coordinate_index, _raise_module_load_error, validate_threads
from ..lazykde import LazyKDE
from ._io_utils import (
    _bin_coordinates,
    _categorical_coordinate,
    _filter_genes,
    _open_file,
)
from ._stereoseq_chips import CHIP_RESOLUTION

if TYPE_CHECKING:
    from spatialdata import SpatialData


# Stereo-seq (and other GEM data)

# according to specification the name should be MIDCount
# however the datasets published in the original Stereo-seq publication used
# various names for the count column
_Gem_Count_ColName = Literal["MIDCount", "MIDCounts", "UMICount"]
_count_column_options = get_args(_Gem_Count_ColName)

_GEM_SCHEMA_OVERRIDE = {
    "geneID": pl.Categorical,
    "geneName": pl.Categorical,
    "x": pl.UInt32,
    "y": pl.UInt32,
    "ExonCount": pl.UInt32,
    "CellID": pl.UInt32,
} | {name: pl.UInt32 for name in _count_column_options}


def read_gem_header(filepath: _PathLike) -> dict[str, str]:
    """
    Read the metadata from the top of the GEM file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the GEM file.

    Returns
    -------
    dict[str, str]
    """
    header = dict()
    with _open_file(filepath, "rb") as f:
        for line in f:
            assert isinstance(line, bytes)
            if not line.startswith(b"#"):
                break
            key, value = line.decode().strip("#\n").split("=", 1)
            header[key] = value

    return header


def _get_resolution_from_chip(chip_name: str) -> int | None:
    # sort prefixes by length, otherwise if testing a shorter prefix first
    # which is a subprefix of another one it might return wrong result
    chip_prefixes = sorted(CHIP_RESOLUTION.keys(), key=len, reverse=True)
    for chip_prefix in chip_prefixes:
        if chip_name.startswith(chip_prefix):
            return CHIP_RESOLUTION[chip_prefix]


def _get_gem_resolution(gem_header: dict[str, str]) -> int | None:
    # The field name according to spec is 'Stereo-seqChip' but in the datasets published
    # in the original publication of Stereo-seq we also find 'StereoChip'
    for field in ["Stereo-seqChip", "StereoChip"]:
        if (chip := gem_header.get(field)) is not None:
            return _get_resolution_from_chip(chip)


def _prepare_gem_dataframe(
    df: pl.DataFrame, exon_count: bool, gene_name: bool
) -> pl.DataFrame:
    if exon_count:
        count_col = "ExonCount"
    else:
        count_col = [name for name in _count_column_options if name in df.columns][0]

    gene_col = "geneName" if gene_name else "geneID"

    df = df.rename({count_col: "count", gene_col: "gene"})
    return df.select(["gene", "x", "y", "count"])


@validate_threads
def read_gem_file(
    filepath: _PathLike, *, sep: str = "\t", n_threads: int | None = None, **kwargs
) -> pl.DataFrame:
    """
    Read a GEM file into a DataFrame.

    GEM files are used by e.g. Stereo-Seq and Nova-ST.

    The name of the count column should be 'MIDCount', however,
    `MIDCounts` and `UMICount` are supported.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the GEM file.
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading file and processing. If `None` or 0 this will
        default to the number of available CPUs.
    kwargs
        Other keyword arguments will be passed to :py:func:`polars.read_csv`.

    Returns
    -------
    polars.DataFrame

    Raises
    ------
    ValueError
        If count column has an unknown name.
    """

    df = pl.read_csv(
        Path(filepath),
        separator=sep,
        comment_prefix="#",
        schema_overrides=_GEM_SCHEMA_OVERRIDE,
        n_threads=n_threads,
        **kwargs,
    )

    if not any(name in df.columns for name in _count_column_options):
        raise ValueError(
            f"Unknown count column, one of {_count_column_options} must be present."
        )

    return df


@validate_threads
def read_StereoSeq(
    filepath: _PathLike,
    *,
    resolution: float | None = None,
    gene_name: bool = False,
    exon_count: bool = False,
    sep: str = "\t",
    n_threads: int | None = None,
    **kwargs,
) -> LazyKDE:
    """
    Read a Stereo-seq GEM file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Stereo-seq file.
    resolution : float, optional
        Center-to-center distance of Stere-seq beads in nm, if None
        it will try to detect it from the chip definition in the file header
        if one exists.
    gene_name : bool
        If True will use the 'geneName' column otherwise the 'geneID' column is used
        as gene identifier.
    exon_count : bool
        If True will use the 'ExonCount' column otherwise the 'MIDCount' column is used
        as counts.
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading file and processing. If `None` or 0 this will
        default to the number of available CPUs.
    kwargs
        Other keyword arguments will be passed to :py:func:`polars.read_csv`.

    Returns
    -------
    sainsc.LazyKDE
    """

    df = read_gem_file(filepath, sep=sep, n_threads=n_threads, **kwargs)
    df = _prepare_gem_dataframe(df, exon_count=exon_count, gene_name=gene_name)

    if resolution is None:
        resolution = _get_gem_resolution(read_gem_header(filepath))

    return LazyKDE.from_dataframe(df, resolution=resolution, n_threads=n_threads)


# Xenium
_XENIUM_COLUMNS = {"feature_name": "gene", "x_location": "x", "y_location": "y"}
XENIUM_CTRLS = [
    "^BLANK",
    "^DeprecatedCodeword",
    "^Intergenic",
    "^NegControl",
    "^UnassignedCodeword",
]
"""Patterns for Xenium controls"""


@validate_threads
def read_Xenium(
    filepath: _PathLike,
    *,
    binsize: float = 0.5,
    remove_features: Collection[str] = XENIUM_CTRLS,
    n_threads: int | None = None,
) -> LazyKDE:
    """
    Read a Xenium transcripts file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Xenium transcripts file. Both, csv.gz and parquet files, are supported.
    binsize : float, optional
        Size of each bin in um.
    remove_features : collections.abc.Collection[str], optional
        List of regex patterns to filter the 'feature_name' column,
        :py:attr:`sainsc.io.XENIUM_CTRLS` by default.
        For Xenium v3 parquet files the data is automatically filtered with the
        'is_gene' column, as well.
    n_threads : int | None, optional
        Number of threads used for reading file and processing. If `None` or 0 this will
        default to the number of available CPUs.

    Returns
    -------
    sainsc.LazyKDE
    """
    filepath = Path(filepath)
    columns = list(_XENIUM_COLUMNS.keys())

    if filepath.suffix == ".parquet":
        transcripts = pl.scan_parquet(filepath)

        # 'is_gene' column only exists for Xenium v3 which only has .parquet
        if "is_gene" in transcripts.collect_schema().names():
            transcripts = transcripts.filter(pl.col("is_gene"))

        transcripts = (
            transcripts.select(columns)
            .with_columns(pl.col("feature_name").cast(pl.Categorical))
            .collect()
        )
    else:
        transcripts = pl.read_csv(
            filepath,
            columns=columns,
            schema_overrides={"feature_name": pl.Categorical},
            n_threads=n_threads,
        )

    transcripts = transcripts.rename(_XENIUM_COLUMNS)
    transcripts = _filter_genes(transcripts, remove_features)

    return LazyKDE.from_dataframe(
        transcripts, binsize=binsize, resolution=1_000, n_threads=n_threads
    )


# Vizgen

_VIZGEN_COLUMNS = {"gene": "gene", "global_x": "x", "global_y": "y"}
VIZGEN_CTRLS = ["^Blank"]
"""Patterns for Vizgen controls"""


@validate_threads
def read_Vizgen(
    filepath: _PathLike,
    *,
    binsize: float = 0.5,
    remove_genes: Collection[str] = VIZGEN_CTRLS,
    n_threads: int | None = None,
) -> LazyKDE:
    """
    Read a Vizgen transcripts file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Vizgen transcripts file.
    binsize : float, optional
        Size of each bin in um.
    remove_genes : collections.abc.Collection[str], optional
        List of regex patterns to filter the 'gene' column,
        :py:attr:`sainsc.io.VIZGEN_CTRLS` by default.
    n_threads : int | None, optional
        Number of threads used for reading file and processing. If `None` or 0 this will
        default to the number of available CPUs.

    Returns
    -------
    sainsc.LazyKDE
    """

    transcripts = pl.read_csv(
        Path(filepath),
        columns=list(_VIZGEN_COLUMNS.keys()),
        schema_overrides={"gene": pl.Categorical},
        n_threads=n_threads,
    ).rename(_VIZGEN_COLUMNS)

    transcripts = _filter_genes(transcripts, remove_genes)

    return LazyKDE.from_dataframe(
        transcripts, binsize=binsize, resolution=1_000, n_threads=n_threads
    )


# Binned data


@validate_threads
def read_StereoSeq_bins(
    filepath: _PathLike,
    bin_size: int = 50,
    *,
    spatialdata: bool = False,
    resolution: float | None = None,
    gene_name: bool = False,
    exon_count: bool = False,
    sep: str = "\t",
    n_threads: int | None = None,
    **kwargs,
) -> "AnnData | SpatialData":
    """
    Read a Stereo-seq GEM file into bins.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Stereo-seq file.
    bin_size : int, optional
        Defines the size of bins along both dimensions
        e.g 50 will results in bins of size 50x50.
    spatialdata : bool, optional
        If True will load the data as a SpatialData object else as an AnnData object.
    resolution : float, optional
        Center-to-center distance of Stere-seq beads in nm, if None
        it will try to detect it from the chip definition in the file header
        if one exists.
    gene_name : bool
        If True will use the 'geneName' column otherwise the 'geneID' column is used
        as gene identifier.
    exon_count : bool
        If True will use the 'ExonCount' column otherwise the 'MIDCount' column is used
        as counts.
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading file and processing. If `None` or 0 this will
        default to the number of available CPUs.
    kwargs
        Other keyword arguments will be passed to :py:func:`polars.read_csv`.

    Returns
    -------
    anndata.AnnData | spatialdata.SpatialData
        AnnData or SpatialData object of the bins with coordinates stored in
        :py:attr:`anndata.AnnData.obsm` with the key `'spatial'`.

    Raises
    ------
    ModuleNotFoundError
        If `spatialdata` is set to `True` but the package is not installed.
    """
    df = read_gem_file(filepath, sep=sep, n_threads=n_threads, **kwargs)
    df = _prepare_gem_dataframe(df, exon_count=exon_count, gene_name=gene_name)
    df = _bin_coordinates(df, bin_size)

    coord_codes, coordinates = _categorical_coordinate(
        df["x"].to_numpy(), df["y"].to_numpy(), n_threads=n_threads
    )
    df = df.drop(["x", "y"])

    genes = pd.DataFrame(index=pd.Index(df["gene"].cat.get_categories(), name="gene"))

    # Duplicate entries in csr_matrix are summed which automatically gives bin merging
    counts = csr_matrix(
        (df["count"], (coord_codes, df["gene"].to_physical())),
        shape=(coordinates.shape[0], df["gene"].cat.get_categories().shape[0]),
        dtype=np.int32,
    )

    del coord_codes, df

    obs = pd.DataFrame(
        index=_get_coordinate_index(
            coordinates[:, 0], coordinates[:, 1], name="bin", n_threads=n_threads
        )
    )

    header = read_gem_header(filepath)

    if resolution is None:
        resolution = _get_gem_resolution(header)

    adata = AnnData(
        X=counts,
        obs=obs,
        var=genes,
        obsm={"spatial": coordinates},
        uns={"file_header": header, "resolution": resolution, "bin_size": bin_size},
    )

    if spatialdata:
        try:
            from geopandas import GeoDataFrame
            from shapely import Polygon
            from spatialdata import SpatialData
            from spatialdata.models import ShapesModel, TableModel

            bin_name = f"bins{bin_size}"

            x, y = adata.obsm["spatial"].T
            del adata.obsm["spatial"]

            df = pd.DataFrame({"x": x, "y": y}, index=adata.obs_names).assign(
                x1=lambda df: df["x"] * bin_size,
                x2=lambda df: df["x1"] + bin_size,
                y1=lambda df: df["y"] * bin_size,
                y2=lambda df: df["y1"] + bin_size,
            )

            shapes = ShapesModel.parse(
                GeoDataFrame(
                    {
                        "geometry": df.apply(
                            lambda r: Polygon(
                                [
                                    (r["x1"], r["y1"]),
                                    (r["x1"], r["y2"]),
                                    (r["x2"], r["y2"]),
                                    (r["x2"], r["y1"]),
                                ]
                            ),
                            axis=1,
                        )
                    }
                )
            )

            adata.obs["region"] = bin_name
            adata.obs["region"] = adata.obs["region"].astype("category")
            adata.obs["instance_key"] = adata.obs_names
            table = TableModel.parse(
                adata, region=bin_name, region_key="region", instance_key="instance_key"
            )

            return SpatialData(
                shapes={bin_name: shapes}, tables={f"{bin_name}_annotation": table}
            )

        except ModuleNotFoundError as e:
            _raise_module_load_error(
                e, "read_StereoSeq_bins", pkg="spatialdata", extra="spatialdata"
            )

    else:
        return adata
