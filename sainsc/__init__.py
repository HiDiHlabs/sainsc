from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sainsc")
except PackageNotFoundError:
    __version__ = "unknown version"

from ._utils_rust import GridCounts, GridFloats
from .io import read_StereoSeq, read_StereoSeq_bins
from .lazykde import LazyKDE, gaussian_kernel

__all__ = [
    "GridCounts",
    "GridFloats",
    "LazyKDE",
    "gaussian_kernel",
    "read_StereoSeq",
    "read_StereoSeq_bins",
]
