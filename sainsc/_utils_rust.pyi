from abc import ABC
from typing import Any

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from ._typealias import _Csx, _CsxArray

def sparse_kde_csxf32(
    counts: _Csx, kernel: NDArray[np.float32], *, threshold: float = 0
) -> _CsxArray:
    """
    Calculate the KDE for each spot with values as float32.
    """

def sparse_kde_csxu32(
    counts: _Csx, kernel: NDArray[np.float32], *, threshold: float = 0
) -> _CsxArray:
    """
    Calculate the KDE for each spot with counts as uint32.
    """

def gridcounts_kde_at_coord(
    counts: GridCounts,
    genes: list[str],
    kernel: NDArray[np.float32],
    coordinates: tuple[NDArray[np.int_], NDArray[np.int_]],
    *,
    n_threads: int | None = None,
) -> _CsxArray:
    """
    Calculate KDE at the given coordinates.
    """

def gridfloats_kde_at_coord(
    counts: GridFloats,
    genes: list[str],
    kernel: NDArray[np.float32],
    coordinates: tuple[NDArray[np.int_], NDArray[np.int_]],
    *,
    n_threads: int | None = None,
) -> _CsxArray:
    """
    Calculate KDE at the given coordinates.
    """

def categorical_coordinate(
    x: NDArray[np.int32], y: NDArray[np.int32], *, n_threads: int | None = None
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Get the codes and the coordinates (comparable to a pandas.Categorical)
    """

def coordinate_as_string(
    x: NDArray[np.int32], y: NDArray[np.int32], *, n_threads: int | None = None
) -> NDArray[np.str_]:
    """
    Concatenate two int arrays elementwise into a string representation (i.e. 'x_y').
    """

def gridcounts_cosinef32_celltypei8(
    counts: GridCounts,
    genes: list[str],
    signatures: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    log: bool = False,
    low_memory: bool = True,
    chunk_size: tuple[int, int] = (500, 500),
    n_threads: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int8]]:
    """
    Calculate the cosine similarity given counts and signatures and assign the most
    similar celltype.
    """

def gridcounts_cosinef32_celltypei16(
    counts: GridCounts,
    genes: list[str],
    signatures: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    log: bool = False,
    low_memory: bool = True,
    chunk_size: tuple[int, int] = (500, 500),
    n_threads: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int16]]:
    """
    Calculate the cosine similarity given counts and signatures and assign the most
    similar celltype.
    """

def gridfloats_cosinef32_celltypei8(
    counts: GridFloats,
    genes: list[str],
    signatures: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    log: bool = False,
    low_memory: bool = True,
    chunk_size: tuple[int, int] = (500, 500),
    n_threads: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int8]]:
    """
    Calculate the cosine similarity given counts and signatures and assign the most
    similar celltype.
    """

def gridfloats_cosinef32_celltypei16(
    counts: GridFloats,
    genes: list[str],
    signatures: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    log: bool = False,
    low_memory: bool = True,
    chunk_size: tuple[int, int] = (500, 500),
    n_threads: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int16]]:
    """
    Calculate the cosine similarity given counts and signatures and assign the most
    similar celltype.
    """

class GridFeatures(ABC):
    """
    Abstract Base class to avoid duplication of documentation
    """

    shape: tuple[int, int]
    """
    tuple[int, int]: Shape of the count arrays.
    """

    def __init__(
        self,
        counts: dict[str, _Csx],
        *,
        resolution: float | None = None,
        n_threads: int | None = None,
    ):
        """
        Parameters
        ----------
        counts : dict[str, scipy.sparse.csr_array | scipy.sparse.csr_matrix | scipy.sparse.csc_array | scipy.sparse.csc_matrix]
            Gene counts.
        resolution : float, optional
            Resolution as nm / pixel.
        n_threads : int, optional
            Number of threads used for reading and processing file. If `None` this will
            default to the number of logical CPUs.

        Raises
        ------
        ValueError
            If genes in `counts` do not all have the same shape.
        """

    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        *,
        resolution: float | None = None,
        binsize: float | None = None,
        n_threads: int | None = None,
    ):  # -> Self
        """
        Initialize from dataframe.

        Generate an instance of this class by transforming a :py:class:`polars.DataFrame`
        that provides a 'gene', 'x', and 'y' column. If a 'count' column exists it is
        used as counts else a count of 1 (single molecule) per row is assumed.

        Parameters
        ----------
        df : polars.DataFrame
            The data to be transformed.
        binsize : float or None, optional
            The size to bin the coordinates by. If None coordinates must be integers.
        resolution : float, optional
            Resolution of each coordinate unit in nm. The default is 1,000 i.e. measurements
            are in um.
        n_threads : int, optional
            Number of threads used for initializing :py:class:`sainsc.LazyKDE`.
            If `None` this will default to the number of logical CPUs.
        """

    def __getitem__(self, key: str) -> _CsxArray: ...
    def __setitem__(self, key: str, value: _Csx): ...
    def __delitem__(self, key: str): ...
    def __len__(self) -> int: ...
    def __contains__(self, item: str) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def get(self, key: str, default: _CsxArray | None = None) -> _CsxArray | None:
        """
        Get the counts for a gene.

        Parameters
        ----------
        key : str
            Name of the gene to retrieve.

        Returns
        -------
        scipy.sparse.csr_array | scipy.sparse.csc_array | None
        """

    def genes(self) -> list[str]:
        """
        Get all available genes.

        Returns
        -------
        list[str]
        """

    def gene_counts(self) -> dict[str, Any]:
        """
        Number of counts per gene.

        Returns
        -------
        dict[str, typing.Any]
            Mapping from gene to number of counts.
        """

    def grid_counts(self) -> NDArray:
        """
        Counts per pixel.

        Aggregates counts across all genes.

        Returns
        -------
        numpy.ndarray
        """

    def select_genes(self, genes: set[str]):
        """
        Keep selected genes.

        Parameters
        ----------
        genes : set[str]
            List of gene names to keep.
        """

    def filter_genes_by_count(self, min, max): ...
    def crop(self, x: tuple[int | None, int | None], y: tuple[int | None, int | None]):
        """
        Crop the field of view for all genes.

        Parameters
        ----------
        x : tuple[int | None, int | None]
            Range to crop as `(xmin, xmax)`
        y : tuple[int | None, int | None]
            Range to crop as `(ymin, ymax)`
        """

    def filter_mask(self, mask: NDArray[np.bool_]):
        """
        Filter all genes with a binary mask.

        Parameters
        ----------
        mask : numpy.ndarray[numpy.bool]
            All counts where `mask` is `False` will be set to 0.
        """

    @property
    def resolution(self) -> float | None:
        """
        float | None: Resolution in nm / pixel.

        Raises
        ------
            TypeError
                If setting with a type other than `float` or `int`.
        """

    @resolution.setter
    def resolution(self, resolution: float): ...
    @property
    def n_threads(self) -> int:
        """
        int: Number of threads used for processing.

        Raises
        ------
            TypeError
                If setting with a type other than `int` or less than 0.
        """

    @n_threads.setter
    def n_threads(self, n_threads: int): ...

class GridCounts(GridFeatures):
    """
    Object holding each gene as count data in a sparse 2D-grid.
    """

    def filter_genes_by_count(self, min: int = 0, max: int = np.iinfo(np.uint32).max):
        """
        Filter genes by minimum and maximum count thresholds.

        Parameters
        ----------
        min : int, optional
            Minimum count threshold.
        max : int, optional
            Maximum count threshold.
        """

    def gene_counts(self) -> dict[str, int]:
        """
        Number of counts per gene.

        Returns
        -------
        dict[str, int]
            Mapping from gene to number of counts.
        """

    def grid_counts(self) -> NDArray[np.uintc]:
        """
        Counts per pixel.

        Aggregates counts across all genes.

        Returns
        -------
        numpy.ndarray[numpy.uintc]
        """

class GridFloats(GridFeatures):
    """
    Object holding each gene in a sparse 2D-grid.
    """

    def filter_genes_by_count(
        self, min: float = 0, max: float = float(np.finfo(np.float32).max)
    ):
        """
        Filter genes by minimum and maximum count thresholds.

        Parameters
        ----------
        min : float, optional
            Minimum count threshold.
        max : float, optional
            Maximum count threshold.
        """

    def gene_counts(self) -> dict[str, float]:
        """
        Number of counts per gene.

        Returns
        -------
        dict[str, float]
            Mapping from gene to number of counts.
        """

    def grid_counts(self) -> NDArray[np.float32]:
        """
        Counts per pixel.

        Aggregates counts across all genes.

        Returns
        -------
        numpy.ndarray[numpy.single]
        """
