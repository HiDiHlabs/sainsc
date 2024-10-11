import math
from typing import TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy import ndimage, signal

T = TypeVar("T", bound=np.number)


def _make_circular_mask(radius: int) -> NDArray[np.bool_]:
    diameter = radius * 2 + 1
    x, y = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    return dist_from_center <= radius


def _make_circular_kernel(kernel: NDArray[T], radius: int) -> NDArray[T]:
    kernel[~_make_circular_mask(radius)] = 0
    return kernel


def gaussian_kernel(
    bw: float, radius: int, *, dtype: DTypeLike = np.float32, circular: bool = False
) -> NDArray:
    """
    Generate a 2D Gaussian kernel array.

    Parameters
    ----------
    bw : float
        Bandwidth of the Gaussian.
    radius : int
        Radius of the kernel. Output size will be :math:`2*radius+1`.
    dtype : numpy.typing.DTypeLike
        Datatype of the kernel.
    circular : bool, optional
        Whether to make kernel circular. Values outside `radius` will be set to 0.

    Returns
    -------
    numpy.ndarray
    """
    mask_size = 2 * radius + 1

    dirac = signal.unit_impulse((mask_size, mask_size), idx="mid")

    gaussian_kernel = ndimage.gaussian_filter(
        dirac, bw, output=np.float64, radius=radius
    ).astype(dtype)

    if circular:
        gaussian_kernel = _make_circular_kernel(gaussian_kernel, radius)

    return gaussian_kernel


def epanechnikov_kernel(bw: float, *, dtype: DTypeLike = np.float32) -> np.ndarray:
    """
    Generate a 2D Epanechnikov kernel array.

    :math:`K(x) = 1/2 * c_d^{-1}*(d+2)(1-||x||^2)` if :math:`||x|| < 1` else 0,
    where :math:`d` is the number of dimensions
    and :math:`c_d` the volume of the unit `d`-dimensional sphere.

    Parameters
    ----------
    bw : float
        Bandwidth of the kernel.
    dtype : numpy.typing.DTypeLike, optional
        Datatype of the kernel.

    Returns
    -------
    numpy.ndarray
    """
    # https://doi.org/10.1109/CVPR.2000.854761
    # c_d = pi for d=2

    r = math.ceil(bw)
    dia = 2 * r - 1  # values at r are zero anyways so the kernel matrix can be smaller

    # 1/2 * pi^-1 * (d+2)
    scale = 2 / math.pi

    kernel = np.zeros((dia, dia), dtype=dtype)
    for i in range(dia):
        for j in range(dia):
            x = i - r + 1
            y = j - r + 1
            norm = (x / bw) ** 2 + (y / bw) ** 2
            if norm < 1:
                kernel[i, j] = scale * (1 - norm)

    return kernel / np.sum(kernel)
