from typing import TYPE_CHECKING

import pandas as pd

from . import __version__
from ._utils import _raise_module_load_error

if TYPE_CHECKING:
    from pooch import Pooch

# version tags have the format vX.Y.Z but __version__ is X.Y.Z
# we need to modify it otherwise the url is incorrect
version = "v" + __version__


def _get_signature_pooch() -> "Pooch":
    # use indirection to enable pooch as optional dependency w/o lazy loading
    try:
        import pooch

        SIGNATURES = pooch.create(
            path=pooch.os_cache("sainsc"),
            base_url="https://github.com/HiDiHLabs/sainsc/raw/{version}/data/",
            version=version,
            version_dev="main",
            registry={
                "signatures_brain.tsv": "sha256:1e7e3e959ea0a0efdfb8bff2ef1de368757a26f317088122a14dd0b141f7149e",
                "signatures_embryo.tsv": "sha256:8554959144fa2e38b2ce0fa59bd4911d09ef84e0565917371bc7a16e062ad68a",
                "embryo_selection.png": "sha256:431011086d5c537c691d2598611d5ea048c0d0808186ba3f924cd7deb625ef56",
            },
        )
        return SIGNATURES
    except ModuleNotFoundError as e:
        _raise_module_load_error(e, "fetch_*_signatures", pkg="pooch", extra="example")


def fetch_brain_signatures() -> pd.DataFrame:
    """
    Load the Stereo-seq mouse hemibrain cell-type signatures.
    """
    fname = _get_signature_pooch().fetch("signatures_brain.tsv")
    data = pd.read_table(fname, index_col=0)
    return data


def fetch_embryo_signatures() -> pd.DataFrame:
    """
    Load the Stereo-seq embryo cell-type signatures.
    """
    fname = _get_signature_pooch().fetch("signatures_embryo.tsv")
    data = pd.read_table(fname, index_col=0)
    return data


def fetch_embryo_mask() -> str:
    """
    Load the Stereo-seq embryo selection mask image.
    """
    fname = _get_signature_pooch().fetch("embryo_selection.png")
    return fname
