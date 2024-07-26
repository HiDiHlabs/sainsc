import importlib.metadata
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sainsc"
copyright = f"""
{datetime.now():%Y}, Niklas Müller-Bötticher, Naveed Ishaque, Roland Eils,
Berlin Institute of Health @ Charité"""
author = "Niklas Müller-Bötticher"
version = importlib.metadata.version("sainsc")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "myst_nb",
]

nb_execution_mode = "off"


autodoc_typehints = "none"
autodoc_typehints_format = "short"

autoapi_dirs = ["../../sainsc"]
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"
autoapi_own_page_level = "attribute"
autoapi_template_dir = "_templates"
autoapi_member_order = "groupwise"

python_use_unqualified_type_names = True  # still experimental

autosummary_generate = True
autosummary_imported_members = True

nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.typing.DTypeLike"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:mod", "polars"),
    ("py:class", "polars.DataFrame"),
    ("py:class", "optional"),
]

exclude_patterns: list[str] = ["_templates"]

intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    polars=("https://docs.pola.rs/py-polars/html/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    spatialdata=("https://spatialdata.scverse.org/en/stable/", None),
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []


def skip_submodules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)
