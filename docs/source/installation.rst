Installation
============


PyPI and ``pip``
----------------

``sainsc`` will soon be available to install from `PyPI <https://pypi.org/>`_.

.. To install ``sainsc`` from `PyPI <https://pypi.org/>`_ using ``pip`` just run

.. .. code-block:: bash

..     pip install sainsc

.. If you want to have support for :py:mod:`spatialdata` use

.. .. code-block:: bash

..     pip install 'sainsc[spatialdata]'


Bioconda and ``conda``
----------------------

``sainsc`` is not yet available for
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installations. But we are
planning to add it to the `bioconda <https://bioconda.github.io/>`_ channel soon.


.. Alternatively, if you prefer the installation using
.. `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ you can do that from the
.. `bioconda <https://bioconda.github.io/>`_ channel.

.. .. code-block:: bash

..     conda install -c bioconda sainsc

.. .. note::

..     Of course, it is also possible to use ``mamba`` instead of ``conda``
..     to speed up the installation.


From GitHub
-----------

You can install the latest versions directly from
[GitHub](https://github.com/HiDiHlabs/sainsc). To do so clone the repository using the
``git clone`` command. Navigate into the downloaded directory and install using

.. code-block:: bash

    pip install .

If you want to install the development version you can install the additional optional
dependencies with

.. code-block:: bash

    pip install -e '.[dev]'
