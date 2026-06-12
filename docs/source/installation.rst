Installation
============


PyPI and ``pip``
----------------

To install ``sainsc`` from `PyPI <https://pypi.org/project/sainsc/>`_ using ``pip``
just run

.. code-block:: bash

    pip install sainsc

If you want to have support for :py:mod:`spatialdata` use

.. code-block:: bash

    pip install sainsc[spatialdata]


Bioconda and ``conda``
----------------------

If you prefer the installation using
`Miniconda <https://docs.anaconda.com/miniconda/>`_ you can install from the
`bioconda <http://bioconda.github.io/recipes/sainsc/README.html>`_ channel.

.. code-block:: bash

    conda install bioconda::sainsc

.. note::

    Of course, it is also possible to use ``mamba`` instead of ``conda``
    to speed up the installation.


From Source
-----------

If you want to to build the package from source (either from GitHub or PyPI) you will need a Rust compiler.
You can follow the `official Rust documentation <https://www.rust-lang.org/tools/install>`_ or,
if you are using ``conda`` install it via ``conda install conda-forge::rust``.

To build the package from PyPI use

.. code-block:: bash

    pip install --no-binary sainsc


GitHub
~~~~~~

You can install the latest versions directly from
`GitHub <https://github.com/HiDiHlabs/sainsc>`_.
To do so clone the repository using the ``git clone`` command.
Navigate into the downloaded directory and install using

.. code-block:: bash

    pip install .

If you want to install the development version you can install the additional optional
dependencies with

.. code-block:: bash

    pip install -e .[dev]


.. warning::

    If you install editable builds (``-e``) the Rust compiler will default to a debug build,
    i.e., the code is not optimized and slow. Don't use this for analysis (only for development)!
