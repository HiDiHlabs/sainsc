from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy.sparse import csr_matrix

from .._typealias import _PathLike
from .._utils import (
    _bin_coordinates,
    _get_coordinate_index,
    _get_n_cpus,
    _raise_module_load_error,
)
from .._utils_rust import GridCounts
from ..lazykde import LazyKDE
from ._io_utils import _categorical_coordinate, _open_file
from ._stereoseq_chips import CHIP_RESOLUTION

if TYPE_CHECKING:
    from spatialdata import SpatialData


# Stereo-seq
def _get_stereo_header(filepath: _PathLike) -> dict[str, str]:
    header = dict()
    with _open_file(filepath, "rb") as f:
        for line in f:
            assert isinstance(line, bytes)
            if not line.startswith(b"#"):
                break
            key, value = line.decode().strip("#\n").split("=", 1)
            header[key] = value

    return header


def _get_stereo_resolution(name: str) -> int | None:
    for chip_name in CHIP_RESOLUTION.keys():
        if name.startswith(chip_name):
            return CHIP_RESOLUTION[chip_name]


def read_gem_file(
    filepath: _PathLike, *, sep: str = "\t", n_threads: int | None = None, **kwargs
) -> pl.DataFrame:
    """
    Read a GEM file into a DataFrame.

    GEM files are used by e.g. Stereo-Seq and Nova-ST.

    The gene-ID and count column will be renamed to `gene` and `count`, respectively.

    The name of the count column must be one of `MIDCounts`, `MIDCount`, or `UMICount`.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the GEM file.
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading and processing file. If `None` this will
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
    _Count_ColName = Literal["MIDCounts", "MIDCount", "UMICount"]

    if n_threads is None:
        n_threads = _get_n_cpus()

    path = Path(filepath)

    columns = pl.read_csv(path, separator=sep, comment_char="#", n_rows=0).columns
    count_col = None
    for name in get_args(_Count_ColName):
        if name in columns:
            count_col = name
            break

    if count_col is None:
        options = get_args(_Count_ColName)
        raise ValueError(
            f"Unknown count column, the name of the column must be one of {options}"
        )
    df = pl.read_csv(
        path,
        separator=sep,
        comment_char="#",
        dtypes={
            "geneID": pl.Categorical,
            "x": pl.Int32,
            "y": pl.Int32,
            count_col: pl.UInt32,
        },
        n_threads=n_threads,
        **kwargs,
    )
    df = df.rename({count_col: "count", "geneID": "gene"})

    return df


def read_StereoSeq(
    filepath: _PathLike,
    *,
    resolution: float | None = None,
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
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading and processing file. If `None` this will
        default to the number of available CPUs.
    kwargs
        Other keyword arguments will be passed to :py:func:`polars.read_csv`.

    Returns
    -------
    sainsc.LazyKDE
    """
    if n_threads is None:
        n_threads = _get_n_cpus()

    if resolution is None:
        chip = _get_stereo_header(filepath).get("StereoChip")
        if chip is not None:
            resolution = _get_stereo_resolution(chip)

    df = read_gem_file(filepath, sep=sep, n_threads=n_threads, **kwargs)

    counts = GridCounts.from_dataframe(
        df, binsize=None, resolution=resolution, n_threads=n_threads
    )

    return LazyKDE(counts, n_threads=n_threads)


def read_StereoSeq_bins(
    filepath: _PathLike,
    bin_size: int = 50,
    *,
    spatialdata: bool = False,
    resolution: float | None = None,
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
    sep : str, optional
        Separator used in :py:func:`polars.read_csv`.
    n_threads : int, optional
        Number of threads used for reading and processing file. If `None` this will
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
    if n_threads is None:
        n_threads = _get_n_cpus()

    header = _get_stereo_header(filepath)
    df = read_gem_file(filepath, sep=sep, n_threads=n_threads, **kwargs)
    df = df.with_columns(pl.col(i) - pl.col(i).min() for i in ["x", "y"])
    df = _bin_coordinates(df.to_pandas(), bin_size)

    coord_codes, coordinates = _categorical_coordinate(
        df.pop("x").to_numpy(), df.pop("y").to_numpy(), n_threads=n_threads
    )

    # Duplicate entries in csr_matrix are summed which automatically gives bin merging
    counts = csr_matrix(
        (df.pop("count"), (coord_codes, df["gene"].cat.codes)),
        shape=(coordinates.shape[0], df["gene"].cat.categories.size),
        dtype=np.int32,
    )

    del coord_codes

    obs = pd.DataFrame(
        index=_get_coordinate_index(
            coordinates[:, 0], coordinates[:, 1], name="bin", n_threads=n_threads
        )
    )
    genes = pd.DataFrame(index=pd.Index(df["gene"].cat.categories, name="gene"))

    if resolution is None:
        chip = header.get("StereoChip")
        if chip is not None:
            resolution = _get_stereo_resolution(chip)

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
