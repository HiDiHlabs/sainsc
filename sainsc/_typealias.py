import os
from typing import TypeAlias

import numpy as np
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

_PathLike: TypeAlias = os.PathLike[str] | str

_Csr: TypeAlias = csr_array | csr_matrix
_Csc: TypeAlias = csc_array | csc_matrix
_Csx: TypeAlias = _Csr | _Csc
_CsxArray: TypeAlias = csc_array | csr_array

_RangeTuple: TypeAlias = tuple[int, int]
_RangeTuple2D: TypeAlias = tuple[_RangeTuple, _RangeTuple]


_Local_Max: TypeAlias = tuple[
    np.ndarray[tuple[int], np.dtype[np.int_]], np.ndarray[tuple[int], np.dtype[np.int_]]
]

_Color: TypeAlias = str | tuple[float, float, float]
_Cmap: TypeAlias = str | list[_Color] | dict[str, _Color]


_Coord: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int32]]
_CosineMap: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_AssignmentScoreMap: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_Kernel: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_SignatureArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_Background: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
_CountMap: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.unsignedinteger]]
_KDE: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.single]]
