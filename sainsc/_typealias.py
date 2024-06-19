import os
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

_PathLike: TypeAlias = os.PathLike[str] | str

_Csr: TypeAlias = csr_array | csr_matrix
_Csc: TypeAlias = csc_array | csc_matrix
_Csx: TypeAlias = _Csr | _Csc
_CsxArray: TypeAlias = csc_array | csr_array

_RangeTuple: TypeAlias = tuple[int, int]
_RangeTuple2D: TypeAlias = tuple[_RangeTuple, _RangeTuple]

_Local_Max: TypeAlias = tuple[NDArray[np.int_], NDArray[np.int_]]

_Color: TypeAlias = str | tuple[float, float, float]
_Cmap: TypeAlias = str | list[_Color] | dict[str, _Color]
