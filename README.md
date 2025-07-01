# sainsc

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

/ˈsaiəns/

_"**S**egmentation-free **A**nalysis of **In S**itu **C**apture data"_
or alternatively
"_**S**tupid **A**cronyms **In Sc**ience_"

`sainsc` is a segmentation-free analysis tool for spatial transcriptomics from in situ
capture technologies (but also works for imaging-based technologies). It is easily
integratable with the [scverse](https://github.com/scverse) (i.e. `scanpy` and `squidpy`)
by exporting data in [`AnnData`](https://anndata.readthedocs.io/) or
[`SpatialData`](https://spatialdata.scverse.org/) format.

## Installation

`sainsc` is available on [PyPI](https://pypi.org/) and [bioconda](https://bioconda.github.io/).

```sh
# PyPI
pip install sainsc
```

```sh
# or conda
conda install bioconda::sainsc
```

For detailed installation instructions please refer to the
[documentation](https://sainsc.readthedocs.io/page/installation.html).

## Documentation

For an extensive documentation of the package please refer to the
[ReadTheDocs page](https://sainsc.readthedocs.io)

## Versioning

This project follows the [SemVer](https://semver.org/) guidelines for versioning.

## Citations

If you are using `sainsc` for your research please cite

N. Müller-Bötticher, S. Tiesmeyer, R. Eils, N. Ishaque, "Sainsc: A Computational Tool
for Segmentation-Free Analysis of In Situ Capture Data" *Small Methods* (2025)
https://doi.org/10.1002/smtd.202401123

```
@article{sainsc2025,
  author = {Müller-Bötticher, Niklas and Tiesmeyer, Sebastian and Eils, Roland and Ishaque, Naveed},
  title = {Sainsc: A Computational Tool for Segmentation-Free Analysis of In Situ Capture Data},
  journal = {Small Methods},
  year = {2025},
  volume = {9},
  number = {5},
  pages = {2401123},
  doi = {10.1002/smtd.202401123},
}
```

## License

This project is licensed under the MIT License - for details please refer to the
[LICENSE](./LICENSE) file.
