from collections.abc import Iterable
from itertools import chain
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits import axes_grid1
from numba import njit
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array
from skimage.feature import peak_local_max
from typing_extensions import Self

from .._typealias import _Cmap, _Csx, _CsxArray, _Local_Max, _RangeTuple2D
from .._utils import _raise_module_load_error, _validate_n_threads, validate_threads
from .._utils_rust import (
    GridCounts,
    cosinef32_and_celltypei8,
    cosinef32_and_celltypei16,
    kde_at_coord,
    sparse_kde_csx_py,
)
from ..utils import gaussian_kernel
from ._utils import (
    SCALEBAR_PARAMS,
    CosineCelltypeCallable,
    _apply_color,
    _filter_blobs,
    _get_cell_dtype,
    _localmax_anndata,
)

# from typing import Self

if TYPE_CHECKING:
    from spatialdata import SpatialData


class LazyKDE:
    """
    Class to analyze kernel density estimates (KDE) for large number of genes.

    The KDE of the genes will be calculated when needed to avoid storing large volumes
    of data in memory.
    """

    @validate_threads
    def __init__(self, counts: GridCounts, *, n_threads: int | None = None):
        """
        Parameters
        ----------
        counts : sainsc.GridCounts
            Gene counts.
        n_threads : int, optional
            Number of threads used for processing. If `None` or 0 this will default to
            the number of available CPUs.
        """

        self.counts: GridCounts = counts
        """
        sainsc.GridCounts :  Spatial gene counts.
        """

        # n_threads is validated (decorator) and will be int
        # but this can currently not be reflected in the type checker
        assert isinstance(n_threads, int)
        self.counts.n_threads = n_threads
        self._threads = n_threads

        self._kernel: NDArray[np.float32] | None = None
        self._total_mRNA: NDArray[np.unsignedinteger] | None = None
        self._total_mRNA_KDE: NDArray[np.float32] | None = None
        self._background: NDArray[np.bool_] | None = None
        self._local_maxima: _Local_Max | None = None
        self._celltype_map: NDArray[np.signedinteger] | None = None
        self._cosine_similarity: NDArray[np.float32] | None = None
        self._assignment_score: NDArray[np.float32] | None = None
        self._celltypes: list[str] | None = None

    @classmethod
    @validate_threads
    def from_dataframe(
        cls, df: pl.DataFrame | pd.DataFrame, *, n_threads: int | None = None, **kwargs
    ) -> Self:
        """
        Construct a LazyKDE from a DataFrame.

        The DataFrame must provide a 'gene', 'x', and 'y' column. If a 'count' column
        exists it will be used as counts else a count of 1 (single molecule) per row
        will be assumed.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
        n_threads : int, optional
            Number of threads used for processing. If `None` or 0 this will default to
            the number of available CPUs.
        kwargs
            Other keyword arguments are passed to
            :py:meth:`sainsc.GridCounts.from_dataframe`.
        """

        count_col = ["count"] if "count" in df.columns else []
        columns = ["gene", "x", "y"] + count_col

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df[columns])
        else:
            df = df.select(pl.col(columns))

        return cls(
            GridCounts.from_dataframe(df, n_threads=n_threads, **kwargs),
            n_threads=n_threads,
        )

    ## Kernel
    def gaussian_kernel(
        self,
        bw: float,
        *,
        unit: str = "px",
        truncate: float = 2,
        circular: bool = False,
    ):
        """
        Set the kernel used for kernel density estimation (KDE) to gaussian.

        Parameters
        ----------
        bw : float
            Bandwidth of the kernel.
        unit : str
            Which unit the bandwidth of the kernel is defined in: 'px' or 'um'.
            'um' requires :py:attr:`sainsc.LazyKDE.resolution` to be set correctly.
        truncate : float, optional
            The radius for calculating the KDE is calculated as `bw` * `truncate`.
            Refer to :py:func:`scipy.ndimage.gaussian_filter`.
        circular : bool, optional
            If `True` calculate the KDE using a circular kernel instead of square by
            setting all values outside the radius `bw` * `truncate` to 0.

        Raises
        ------
        ValueError
            If `unit` is neither 'px' nor 'um'.
        ValueError
            If `unit` is 'um' but `resolution` is not set.

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.kde`
        """

        if unit == "um":
            if self.resolution is None:
                raise ValueError(
                    "Using `unit`='um' requires the `resolution` to be set."
                )
            bw /= self.resolution / 1_000
        elif unit != "px":
            raise ValueError("`unit` must be either 'px' or 'um'")
        dtype = np.float32
        radius = round(truncate * bw)
        self.kernel = gaussian_kernel(bw, radius, dtype=dtype, circular=circular)

    ## KDE
    def kde(self, gene: str, *, threshold: float | None = None) -> _CsxArray:
        """
        Calculate kernel density estimate (KDE).

        The kernel will be used from :py:attr:`sainsc.LazyKDE.kernel`.

        Parameters
        ----------
        gene : collections.abc.Sequence[str]
            List of genes for which to calculate the KDE.
        threshold : float, optional
            Relative threshold of maximum of the kernel that is used to filter beads.
            All values below :math:`threshold * max(kernel)` are set to 0. Filtering is done
            after calculating the KDE which sets it apart from reducing `truncate`.

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.gaussian_kernel`
        """
        return self._kde(self.counts[gene], threshold)

    def _kde(self, arr: NDArray | _Csx, threshold: float | None = None) -> _CsxArray:
        if self.kernel is None:
            raise ValueError("`kernel` must be set before running KDE")

        if threshold is None:
            threshold = 0

        if isinstance(arr, np.ndarray):
            # scipy.ndimage.convolve could be used for dense arrays but seems to be
            # slower than converting to sparse and running custom kde
            arr = csr_array(arr)

        if arr.dtype == np.uint32:
            return sparse_kde_csx_py(arr, self.kernel, threshold=threshold)
        else:
            raise TypeError("Sparse KDE currently only supports 'numpy.uint32'")

    def calculate_total_mRNA(self):
        """
        Calculate kernel density estimate (KDE) for the total mRNA.

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.calculate_total_mRNA_KDE`
        """

        self._total_mRNA = self.counts.grid_counts()

    def calculate_total_mRNA_KDE(self):
        """
        Calculate kernel density estimate (KDE) for the total mRNA.

        If :py:attr:`sainsc.LazyKDE.total_mRNA` has not been calculated
        :py:meth:`sainsc.LazyKDE.calculate_total_mRNA` is run first.

        Raises
        ------
        ValueError
            If `self.kernel` is not set.

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.gaussian_kernel`
        :py:meth:`sainsc.LazyKDE.kde`
        """
        if self.total_mRNA is None or self.total_mRNA.shape != self.shape:
            self.calculate_total_mRNA()
        assert self.total_mRNA is not None
        self._total_mRNA_KDE = self._kde(self.total_mRNA).toarray()

    ## Local maxima / cell proxies
    def find_local_maxima(self, min_dist: int, min_area: int = 0):
        """
        Find the local maxima of the kernel density estimates.

        The local maxima are detected from the KDE of the total mRNA stored in
        :py:attr:`sainsc.LazyKDE.total_mRNA_KDE`. Background as defined in
        :py:attr:`sainsc.LazyKDE.background` will be removed before identifying
        local maxima.

        Parameters
        ----------
        min_dist : int
            Minimum distance between two maxima in pixels.
        min_area : int, optional
            Minimum area of connected pixels that are not background to be
            considered for maxima detection. Allows ignoring maxima in noisy spots.
        """
        if self.total_mRNA_KDE is None:
            raise ValueError(
                "`total_mRNA_KDE` must be calculated before finding local maxima"
            )

        if self.background is not None:
            foreground = ~self.background
            if min_area > 0:
                foreground = _filter_blobs(foreground, min_area)
        else:
            foreground = None

        local_max = peak_local_max(
            self.total_mRNA_KDE,
            min_distance=min_dist,
            exclude_border=False,
            labels=foreground,
        )

        self._local_maxima = (local_max[:, 0], local_max[:, 1])

    def load_local_maxima(
        self, genes: Iterable[str] | None = None, *, spatialdata: bool = False
    ) -> "AnnData | SpatialData":
        """
        Load the gene expression (KDE) of the local maxima.

        The local maxima (:py:attr:`sainsc.LazyKDE.local_maxima`) are calculated and
        returned as :py:class:`anndata.AnnData` object.

        Parameters
        ----------
        genes : collections.abc.Iterable[str], optional
            List of genes for which the KDE will be calculated.
        spatialdata : bool, optional
            If True will load the data as a SpatialData object including the totalRNA
            projection and cell-type map if available. If False an AnnData object is
            returned.

        Returns
        -------
        anndata.AnnData | spatialdata.SpatialData

        Raises
        ------
        ModuleNotFoundError
            If `spatialdata` is set to `True` but the package is not installed.
        ValueError
            If `self.kernel` is not set.
        """
        if self.local_maxima is None:
            raise ValueError("`local_maxima` have to be identified before loading")

        genes = self.genes if genes is None else list(genes)

        kde = self._load_KDE_maxima(genes)
        adata = _localmax_anndata(
            kde, genes, self.local_maxima, name="local_maxima", n_threads=self.n_threads
        )

        if spatialdata:
            try:
                from spatialdata import SpatialData
                from spatialdata.models import (
                    Image2DModel,
                    Labels2DModel,
                    PointsModel,
                    TableModel,
                )

                x, y = adata.obsm["spatial"].T
                del adata.obsm["spatial"]

                localmax_name = "local_maxima"

                local_max = PointsModel.parse(
                    pd.DataFrame({"x": x, "y": y}, index=adata.obs_names)
                )

                adata.obs["region"] = localmax_name
                adata.obs["region"] = adata.obs["region"].astype("category")
                adata.obs["instance_key"] = adata.obs_names

                local_max_anno = TableModel.parse(
                    adata,
                    region=localmax_name,
                    region_key="region",
                    instance_key="instance_key",
                )

                sdata_dict: dict[str, Any] = {
                    localmax_name: local_max,
                    f"{localmax_name}_annotation": local_max_anno,
                }

                if self.total_mRNA_KDE is not None:

                    sdata_dict["total_mRNA"] = Image2DModel.parse(
                        np.atleast_3d(self.total_mRNA_KDE).T, dims=("c", "y", "x")
                    )

                if self.celltype_map is not None:
                    label_name = "celltype_map"

                    labels = self.celltype_map + 1
                    if self.background is not None:
                        labels[self.background] = 0

                    sdata_dict[label_name] = Labels2DModel.parse(
                        labels.T, dims=("y", "x")
                    )

                    obs = pd.DataFrame(
                        {"region": label_name, "instance_key": self.celltypes},
                        index=self.celltypes,
                    ).astype({"region": "category"})

                    sdata_dict[f"{label_name}_annotation"] = TableModel.parse(
                        AnnData(obs=obs),
                        region=label_name,
                        region_key="region",
                        instance_key="instance_key",
                    )

                return SpatialData.from_elements_dict(sdata_dict)

            except ModuleNotFoundError as e:
                _raise_module_load_error(
                    e, "load_local_maxima", pkg="spatialdata", extra="spatialdata"
                )

        else:
            load_attr = [
                "total_mRNA_KDE",
                "cosine_similarity",
                "assignment_score",
                "celltype_map",
            ]
            x, y = adata.obsm["spatial"].T
            for name in load_attr:
                if (attr := getattr(self, name)) is not None:
                    # assert isinstance(attr, np.ndarray)
                    values = attr[x, y]
                    if name == "celltype_map":
                        assert self.celltypes is not None
                        adata.obs["celltype"] = pd.Categorical.from_codes(
                            values, self.celltypes
                        )
                    else:
                        adata.obs[name] = values

            return adata

    def _load_KDE_maxima(self, genes: list[str]) -> csc_array | csr_array:

        assert self.local_maxima is not None
        if self.kernel is None:
            raise ValueError("`kernel` must be set before running KDE")

        return kde_at_coord(
            self.counts, genes, self.kernel, self.local_maxima, n_threads=self.n_threads
        )

    ## Celltyping
    def filter_background(
        self,
        min_norm: float | dict[str, float],
        min_cosine: float | dict[str, float] | None = None,
        min_assignment: float | dict[str, float] | None = None,
    ):
        """
        Define pixels as background.

        If using multiple thresholds (e.g. on norm and cosine similarity) they will be
        combined and pixels are defined as background if they are lower than any of the
        thresholds.

        Parameters
        ----------
        min_norm : float or dict[str, float]
            The threshold for defining background based on
            :py:attr:`sainsc.LazyKDE.total_mRNA_KDE`.
            Either a float which is used as global threshold or a mapping from cell types
            to thresholds. Cell-type assignment is needed for cell type-specific thresholds.
        min_cosine : float or dict[str, float], optional
            The threshold for defining background based on
            :py:attr:`sainsc.LazyKDE.cosine_similarity`. Cell type-specific thresholds
            can be defined as for `min_norm`.
        min_assignment : float or dict[str, float], optional
            The threshold for defining background based on
            :py:attr:`sainsc.LazyKDE.assignment_score`. Cell type-specific thresholds
            can be defined as for `min_norm`.

        Raises
        ------
        ValueError
            If cell type-specific thresholds do not include all cell types or if
            using cell type-specific thresholds before cell type assignment.
        """

        @njit
        def _map_celltype_to_value(
            ct_map: NDArray[np.integer], thresholds: tuple[float, ...]
        ) -> NDArray[np.floating]:
            values = np.zeros(shape=ct_map.shape, dtype=float)
            for i in range(ct_map.shape[0]):
                for j in range(ct_map.shape[1]):
                    if ct_map[i, j] >= 0:
                        values[i, j] = thresholds[ct_map[i, j]]
            return values

        if self.total_mRNA_KDE is None:
            raise ValueError(
                "`total_mRNA_KDE` needs to be calculated before filtering background"
            )

        if isinstance(min_norm, dict):
            if self.celltypes is None or self.celltype_map is None:
                raise ValueError(
                    "Cell type-specific threshold can only be used after cell-type assignment"
                )
            elif not all([ct in min_norm.keys() for ct in self.celltypes]):
                raise ValueError("'min_norm' does not contain all celltypes.")
            idx2threshold = tuple(min_norm[ct] for ct in self.celltypes)
            threshold = _map_celltype_to_value(self.celltype_map, idx2threshold)
            background = self.total_mRNA_KDE < threshold
        else:
            background = self.total_mRNA_KDE < min_norm

        if min_cosine is not None:
            if self.cosine_similarity is None:
                raise ValueError(
                    "Cosine similarity threshold can only be used after cell-type assignment"
                )
            if isinstance(min_cosine, dict):
                if self.celltypes is None or self.celltype_map is None:
                    raise ValueError(
                        "Cell type-specific threshold can only be used after cell-type assignment"
                    )
                elif not all([ct in min_cosine.keys() for ct in self.celltypes]):
                    raise ValueError("'min_cosine' does not contain all celltypes.")
                idx2threshold = tuple(min_cosine[ct] for ct in self.celltypes)
                threshold = _map_celltype_to_value(self.celltype_map, idx2threshold)
                background |= self.cosine_similarity <= threshold
            else:
                background |= self.cosine_similarity <= min_cosine

        if min_assignment is not None:
            if self.assignment_score is None:
                raise ValueError(
                    "Assignment score threshold can only be used after cell-type assignment"
                )
            if isinstance(min_assignment, dict):
                if self.celltypes is None or self.celltype_map is None:
                    raise ValueError(
                        "Cell type-specific threshold can only be used after cell-type assignment"
                    )
                elif not all([ct in min_assignment.keys() for ct in self.celltypes]):
                    raise ValueError("'min_assignment' does not contain all celltypes.")
                idx2threshold = tuple(min_assignment[ct] for ct in self.celltypes)
                threshold = _map_celltype_to_value(self.celltype_map, idx2threshold)
                background |= self.assignment_score <= threshold
            else:
                background |= self.assignment_score <= min_assignment

        self._background = background

    @staticmethod
    def _calculate_cosine_celltype_fn(dtype) -> CosineCelltypeCallable:
        if dtype == np.int8:
            return cosinef32_and_celltypei8
        elif dtype == np.int16:
            return cosinef32_and_celltypei16
        else:
            raise NotImplementedError

    def assign_celltype(
        self,
        signatures: pd.DataFrame,
        *,
        log: bool = False,
        chunk: tuple[int, int] = (500, 500),
    ):
        """
        Calculate the cosine similarity with known cell-type signatures.

        For each bead calculate the cosine similarity with a set of cell-type signatures.
        The cell-type with highest score will be assigned to the corresponding bead.

        Parameters
        ----------
        signatures : pandas.DataFrame
            DataFrame of cell-type signatures. Columns are cell-types and index are genes.
        log : bool
            Whether to log transform the KDE when calculating the cosine similarity.
            This is useful if the gene signatures are derived from log-transformed data.
        chunk : tuple[int, int]
            Size of the chunks for processing. Larger chunks require more memory but
            have less duplicated computation.

        Raises
        ------
        ValueError
            If not all genes of the `signatures` are available.
        ValueError
            If `self.kernel` is not set.
        ValueError
            If `chunk` is smaller than the shape of `self.kernel`.
        """

        if not all(signatures.index.isin(self.genes)):
            raise ValueError(
                "Not all genes in the gene signature are part of this KDE."
            )

        if self.kernel is None:
            raise ValueError("`kernel` must be set before running KDE")

        if not all(s < c for s, c in zip(self.kernel.shape, chunk)):
            raise ValueError("`chunk` must be larger than shape of kernel.")

        dtype = np.float32

        celltypes = signatures.columns.tolist()
        ct_dtype = _get_cell_dtype(len(celltypes))

        # scale signatures to unit norm
        signatures_mat = signatures.to_numpy()
        signatures_mat = (
            signatures_mat / np.linalg.norm(signatures_mat, axis=0)
        ).astype(dtype, copy=False)

        genes = signatures.index.to_list()

        fn = self._calculate_cosine_celltype_fn(ct_dtype)

        self._cosine_similarity, self._assignment_score, self._celltype_map = fn(
            self.counts,
            genes,
            signatures_mat,
            self.kernel,
            log=log,
            chunk_size=chunk,
            n_threads=self.n_threads,
        )
        self._celltypes = celltypes

    ## Plotting
    def _plot_2d(
        self,
        img: NDArray,
        title: str,
        *,
        remove_background: bool = False,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        im_kwargs: dict = dict(),
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
    ) -> Figure:
        if remove_background:
            if self.background is not None:
                img = img.copy()
                img[self.background] = 0
            else:
                raise ValueError("`background` is undefined")

        if crop is not None:
            img = img[tuple(slice(*c) for c in crop)]
        fig, ax = plt.subplots(1, 1)
        assert isinstance(ax, Axes)
        im = ax.imshow(img.T, origin="lower", **im_kwargs)
        ax.set_title(title)

        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax)

        if scalebar:
            self._add_scalebar(ax, **scalebar_kwargs)
        return fig

    def _add_scalebar(self, ax: Axes, **kwargs):
        if self.counts.resolution is None:
            raise ValueError("'resolution' must be set when using scalebar")
        ax.add_artist(ScaleBar(self.counts.resolution, units="nm", **kwargs))

    def plot_genecount_histogram(self, **kwargs) -> Figure:
        """
        Plot a histogram of the counts per gene.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to :py:func:`seaborn.histplot`

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(1, 1)
        assert isinstance(ax, Axes)
        sns.histplot(
            np.fromiter(self.counts.gene_counts().values(), dtype=int),
            log_scale=True,
            ax=ax,
            **kwargs,
        )
        ax.set(xlabel="Counts per gene", ylabel="# genes")
        return fig

    def plot_KDE_histogram(
        self,
        *,
        gene: str | None = None,
        remove_background: bool = False,
        crop: _RangeTuple2D | None = None,
        **kwargs,
    ) -> Figure:
        """
        Plot a histogram of the kernel density estimates.

        Plots either the kernel density estimate (KDE) of the total mRNA
        (:py:attr:`sainsc.LazyKDE.total_mRNA_KDE`) or of a single gene if `gene` is
        provided.

        Parameters
        ----------
        gene : str, optional
            Gene for which the KDE histogram is plotted.
        remove_background : bool, optional
            If `True`, all pixels for which :py:attr:`sainsc.LazyKDE.background` is
            `False` are set to 0.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        kwargs
            Other keyword arguments are passed to :py:func:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.kde`
        """
        name = "total mRNA" if gene is None else gene

        if gene is not None:
            kde = self.kde(gene)
        else:
            if self.total_mRNA_KDE is not None:
                kde = self.total_mRNA_KDE
            else:
                raise ValueError("`total_mRNA_KDE` has not been calculated")

        if remove_background:
            if self.background is not None:
                kde[self.background] = 0
            else:
                raise ValueError("`background` is undefined")

        if crop is not None:
            kde = kde[tuple(slice(*c) for c in crop)]

        fig, ax = plt.subplots(1, 1)
        assert isinstance(ax, Axes)
        ax.hist(coo_array(kde).data, **kwargs)
        ax.set(xlabel=f"KDE of {name}", ylabel="# pixels")
        return fig

    def plot_genecount(
        self,
        *,
        gene: str | None = None,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        im_kwargs: dict = dict(),
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
    ) -> Figure:
        """
        Plot the gene expression counts.

        By default this will plot the :py:attr:`sainsc.LazyKDE.total_mRNA`. If
        `gene` is specified the respective gene will be plotted.

        Parameters
        ----------
        gene : str, optional
            Gene in :py:attr:`sainsc.LazyKDE.genes` to use for plotting.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        scalebar : bool, optional
            If `True`, add a ``matplotlib_scalebar.scalebar.ScaleBar`` to the plot.
        im_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to :py:func:`matplotlib.pyplot.imshow`.
        scalebar_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to ``matplotlib_scalebar.scalebar.ScaleBar``.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If :py:attr:`sainsc.LazyKDE.total_mRNA` has not been calculated.
        """
        if gene is not None:
            img = self.counts[gene].toarray()
        else:
            if self.total_mRNA is not None:
                img = self.total_mRNA
            else:
                raise ValueError(
                    "`total_mRNA` has not been calculated."
                    "Run `calculate_total_mRNA` first."
                )
        title = "total mRNA" if gene is None else gene

        return self._plot_2d(
            img,
            title,
            crop=crop,
            scalebar=scalebar,
            im_kwargs=im_kwargs,
            scalebar_kwargs=scalebar_kwargs,
        )

    def plot_KDE(
        self,
        *,
        gene: str | None = None,
        remove_background: bool = False,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        im_kwargs: dict = dict(),
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
    ) -> Figure:
        """
        Plot the kernel density estimate (KDE).

        By default this will plot the KDE of the total mRNA
        (:py:attr:`sainsc.LazyKDE.total_mRNA_KDE`). If `gene` is specified the
        respective KDE will be computed and plotted.

        Parameters
        ----------
        gene : str, optional
            Gene in :py:attr:`sainsc.LazyKDE.genes` to use for plotting.
        remove_background : bool, optional
            If `True`, all pixels for which :py:attr:`sainsc.LazyKDE.background` is
            `False` are set to 0.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        scalebar : bool, optional
            If `True`, add a ``matplotlib_scalebar.scalebar.ScaleBar`` to the plot.
        im_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to :py:func:`matplotlib.pyplot.imshow`.
        scalebar_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to ``matplotlib_scalebar.scalebar.ScaleBar``.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If :py:attr:`sainsc.LazyKDE.total_mRNA_KDE` has not been calculated.

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.kde`
        """
        if gene is not None:
            img = self.kde(gene).toarray()
        else:
            if self.total_mRNA_KDE is not None:
                img = self.total_mRNA_KDE
            else:
                raise ValueError(
                    "`total_mRNA_KDE` has not been calculated."
                    "Run `calculate_total_mRNA_KDE` first."
                )

        title = "KDE of " + ("total mRNA" if gene is None else gene)

        return self._plot_2d(
            img,
            title,
            remove_background=remove_background,
            crop=crop,
            scalebar=scalebar,
            im_kwargs=im_kwargs,
            scalebar_kwargs=scalebar_kwargs,
        )

    def plot_local_maxima(
        self,
        *,
        crop: _RangeTuple2D | None = None,
        background_kwargs: dict = dict(),
        scatter_kwargs: dict = dict(),
    ) -> Figure:
        """
        Plot the local kernel density estimate maxima.

        Parameters
        ----------
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        background_kwargs : dict, optional
            Keyword arguments that are passed to :py:meth:`sainsc.LazyKDE.plot_KDE`.
        scatter_kwargs : dict, optional
            Keyword arguments that are passed to :py:func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.find_local_maxima`
        """
        if self.local_maxima is None:
            raise ValueError

        x, y = self.local_maxima

        if crop is not None:
            x_min, x_max = crop[0]
            y_min, y_max = crop[1]
            keep = (x >= x_min) & (y >= y_min) & (x < x_max) & (y < y_max)
            x = x[keep] - x_min
            y = y[keep] - y_min

        fig = self.plot_KDE(crop=crop, **background_kwargs)
        fig.axes[0].scatter(x, y, **scatter_kwargs)
        return fig

    def plot_celltype_map(
        self,
        *,
        remove_background: bool = True,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        cmap: _Cmap = "hls",
        background: str | tuple = "black",
        undefined: str | tuple = "grey",
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
        return_img: bool = False,
    ) -> Figure | NDArray[np.uint8]:
        """
        Plot the cell-type annotation.

        Parameters
        ----------
        remove_background : bool, optional
            If `True`, all pixels for which :py:attr:`sainsc.LazyKDE.background` is
            `False` are set to 0.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        scalebar : bool, optional
            If `True`, add a ``matplotlib_scalebar.scalebar.ScaleBar`` to the plot.
        cmap : str or list or dict, optional
            If it is a string it must be the name of a `cmap` that can be used in
            :py:func:`seaborn.color_palette`.
            If it is a list of colors it must have the same length as the number of
            celltypes.
            If it is a dictionary it must be a mapping from celltpye to color. Undefined
            celltypes are plotted according to `undefined`.
            Colors can either be provided as string that can be converted via
            :py:func:`matplotlib.colors.to_rgb` or as ``(r, g, b)``-tuple between 0-1.
        background : str | tuple[float, float, float]
            Color for the background.
        undefined : str | tuple[float, float, float]
            Color used for celltypes without a defined color.
        scalebar_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to ``matplotlib_scalebar.scalebar.ScaleBar``.
        return_img : bool, optional
            Return the cell-type map as 3D-array (x, y, RGB) instead of the Figure.

        Returns
        -------
        matplotlib.figure.Figure | numpy.ndarray[numpy.ubyte]


        See Also
        --------
        :py:meth:`sainsc.LazyKDE.assign_celltype`
        """
        if self.celltypes is None or self.celltype_map is None:
            raise ValueError("celltype assignment missing")

        n_celltypes = len(self.celltypes)

        celltype_map = self.celltype_map.copy()
        if remove_background:
            if self.background is None:
                raise ValueError("Background has not been filtered.")
            else:
                celltype_map[self.background] = -1

        if crop is not None:
            celltype_map = celltype_map[tuple(slice(*c) for c in crop)]

        # shift so 0 will be background
        celltype_map += 1

        if isinstance(cmap, str):
            color_map = sns.color_palette(cmap, n_colors=n_celltypes)
            assert isinstance(color_map, Iterable)
        else:
            if isinstance(cmap, list):
                if len(cmap) != n_celltypes:
                    raise ValueError("You need to provide 1 color per celltype")

            elif isinstance(cmap, dict):
                cmap = [cmap.get(cell, undefined) for cell in self.celltypes]

            color_map = [to_rgb(c) if isinstance(c, str) else c for c in cmap]

        # convert to uint8 to reduce memory of final image
        color_map_int = tuple(
            (np.array(c) * 255).round().astype(np.uint8)
            for c in chain([to_rgb(background)], color_map)
        )
        img = _apply_color(celltype_map.T, color_map_int)

        if return_img:
            return img

        legend_elements = [
            Patch(color=c, label=lbl) for c, lbl in zip(color_map, self.celltypes)
        ]

        fig, ax = plt.subplots()
        assert isinstance(ax, Axes)
        ax.imshow(img, origin="lower")
        ax.legend(
            title="Cell type",
            handles=legend_elements,
            ncols=-(n_celltypes // -20),  # ceildiv
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        if scalebar:
            self._add_scalebar(ax, **scalebar_kwargs)

        return fig

    def plot_cosine_similarity(
        self,
        *,
        remove_background: bool = False,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        im_kwargs: dict = dict(),
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
    ) -> Figure:
        """
        Plot the cosine similarity from cell-type assignment.

        Parameters
        ----------
        remove_background : bool, optional
            If `True`, all pixels for which :py:attr:`sainsc.LazyKDE.background` is
            `False` are set to 0.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        scalebar : bool, optional
            If `True`, add a ``matplotlib_scalebar.scalebar.ScaleBar`` to the plot.
        im_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to :py:func:`matplotlib.pyplot.imshow`.
        scalebar_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to ``matplotlib_scalebar.scalebar.ScaleBar``.

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.assign_celltype`
        """
        if self.cosine_similarity is not None:
            return self._plot_2d(
                self.cosine_similarity,
                "Cosine similarity",
                remove_background=remove_background,
                crop=crop,
                scalebar=scalebar,
                im_kwargs=im_kwargs,
                scalebar_kwargs=scalebar_kwargs,
            )
        else:
            raise ValueError("Cell types have not been assigned")

    def plot_assignment_score(
        self,
        *,
        remove_background: bool = False,
        crop: _RangeTuple2D | None = None,
        scalebar: bool = True,
        im_kwargs: dict = dict(),
        scalebar_kwargs: dict = SCALEBAR_PARAMS,
    ) -> Figure:
        """
        Plot the assignment score from cell-type assignment.

        Parameters
        ----------
        remove_background : bool, optional
            If `True`, all pixels for which :py:attr:`sainsc.LazyKDE.background` is
            `False` are set to 0.
        crop : tuple[tuple[int, int], tuple[int, int]], optional
            Coordinates to crop the data defined as `((xmin, xmax), (ymin, ymax))`.
        scalebar : bool, optional
            If `True`, add a ``matplotlib_scalebar.scalebar.ScaleBar`` to the plot.
        im_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to :py:func:`matplotlib.pyplot.imshow`.
        scalebar_kwargs : dict[str, typing.Any], optional
            Keyword arguments that are passed to ``matplotlib_scalebar.scalebar.ScaleBar``.

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        :py:meth:`sainsc.LazyKDE.assign_celltype`
        """
        if self.assignment_score is not None:
            return self._plot_2d(
                self.assignment_score,
                "Assignment score",
                remove_background=remove_background,
                crop=crop,
                scalebar=scalebar,
                im_kwargs=im_kwargs,
                scalebar_kwargs=scalebar_kwargs,
            )
        else:
            raise ValueError("Cell types have not been assigned")

    ## Attributes
    @property
    def n_threads(self) -> int:
        """
        int: Number of threads that will be used for computations.

        Raises
        ------
            ValueError
                If setting with an `int` less than 0.
        """
        return self._threads

    @n_threads.setter
    def n_threads(self, n_threads: int | None):
        self._threads = _validate_n_threads(n_threads)
        self.counts.n_threads = self._threads

    @property
    def shape(self) -> tuple[int, int]:
        """
        tuple[int, int]: Shape of the sample.
        """
        return self.counts.shape

    @property
    def genes(self) -> list[str]:
        """
        list[str]: List of genes.
        """
        return self.counts.genes()

    @property
    def resolution(self) -> float | None:
        """
        float: Resolution in nm / pixel.

        Raises
        ------
            TypeError
                If setting with a type other than `float` or `int`.
        """
        return self.counts.resolution

    @resolution.setter
    def resolution(self, resolution: float | None):
        self.counts.resolution = resolution

    @property
    def kernel(self) -> np.ndarray | None:
        """
        numpy.ndarray: Map of the KDE of total mRNA.

        Raises
        ------
            ValueError
                If kernel is not a square, 2D :py:class:`numpy.ndarray` of uneven length.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: np.ndarray):
        if (
            len(kernel.shape) != 2
            or kernel.shape[0] != kernel.shape[1]
            or any(i % 2 == 0 for i in kernel.shape)
        ):
            raise ValueError(
                "`kernel` currently only supports 2D squared arrays of uneven length."
            )
        else:
            self._kernel = kernel.astype(np.float32)

    @property
    def local_maxima(self) -> _Local_Max | None:
        """
        tuple[numpy.ndarray[numpy.signedinteger], ...]: Coordinates of local maxima.
        """
        return self._local_maxima

    @property
    def total_mRNA(self) -> NDArray[np.unsignedinteger] | None:
        """
        numpy.ndarray[numpy.unsignedinteger]: Map of the total mRNA.
        """
        return self._total_mRNA

    @property
    def total_mRNA_KDE(self) -> NDArray[np.single] | None:
        """
        numpy.ndarray[numpy.single]: Map of the KDE of total mRNA.
        """
        return self._total_mRNA_KDE

    @property
    def background(self) -> NDArray[np.bool_] | None:
        """
        numpy.ndarray[numpy.bool]: Map of pixels that are assigned as background.

        Raises
        ------
            TypeError
                If setting with array that is not of type `numpy.bool`.
            ValueError
                If setting with array that has different shape than `self`.
        """
        return self._background

    @background.setter
    def background(self, background: NDArray[np.bool_]):
        if background.shape != self.shape:
            raise ValueError("`background` must have same shape as `self`")
        else:
            self._background = background

    @property
    def celltypes(self) -> list[str] | None:
        """
        list[str]: List of assigned celltypes.
        """
        return self._celltypes

    @property
    def cosine_similarity(self) -> NDArray[np.single] | None:
        """
        numpy.ndarray[numpy.single]: Cosine similarity for each pixel.
        """
        return self._cosine_similarity

    @property
    def assignment_score(self) -> NDArray[np.single] | None:
        """
        numpy.ndarray[numpy.single]: Assignment score for each pixel.

        Let `x` be the gene expression of a pixel, and `i` and `j` the signatures of the
        best and 2nd best scoring cell type, respectively. The assignment score is
        calculated as :math:`\\frac{cos(\\theta_{xi}) - cos(\\theta_{xj})}{cos(\\pi/2 - \\theta_{ij})}`
        where :math:`\\theta` is the angle between the corresponding vectors.
        """
        return self._assignment_score

    @property
    def celltype_map(self) -> NDArray[np.signedinteger] | None:
        """
        numpy.ndarray[numpy.signedinteger]: Cell-type map of cell-type indices.

        Each number corresponds to the index in :py:attr:`sainsc.LazyKDE.celltypes`,
        and -1 to unassigned (background).
        """
        return self._celltype_map

    def __repr__(self) -> str:
        repr = [
            f"LazyKDE ({self.n_threads} threads)",
            f"genes: {len(self.genes)}",
            f"shape: {self.shape}",
        ]
        if self.resolution is not None:
            repr.append(f"resolution: {self.resolution:.1f} nm / px")
        if self.kernel is not None:
            repr.append(f"kernel: {self.kernel.shape}")
        if self.background is not None:
            repr.append("background: set")
        if self.local_maxima is not None:
            repr.append(f"local maxima: {len(self.local_maxima[0])}")
        if self.celltypes is not None:
            repr.append(f"celltypes: {len(self.celltypes)}")

        spacing = "    "

        return f"\n{spacing}".join(repr)
