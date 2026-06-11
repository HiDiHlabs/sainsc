"""
This module contains functionality supporting reading data of different
spatially-resolved transcriptomics technologies and file formats.
"""

from ._io import (
    ATERA_CTRLS,
    VIZGEN_CTRLS,
    XENIUM_CTRLS,
    read_Atera,
    read_gem_file,
    read_gem_header,
    read_StereoSeq,
    read_StereoSeq_bins,
    read_VisiumHD,
    read_Vizgen,
    read_Xenium,
)

__all__ = [
    "ATERA_CTRLS",
    "VIZGEN_CTRLS",
    "XENIUM_CTRLS",
    "read_Atera",
    "read_gem_file",
    "read_gem_header",
    "read_StereoSeq",
    "read_StereoSeq_bins",
    "read_VisiumHD",
    "read_Vizgen",
    "read_Xenium",
]
