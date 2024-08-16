from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sainsc")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version

from ._utils_rust import GridCounts
from .lazykde import LazyKDE

__all__ = ["GridCounts", "LazyKDE"]
