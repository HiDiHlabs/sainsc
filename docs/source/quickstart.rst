Quickstart / tl;dr
==================

The general workflow will look like this:

You first read your data into an :py:class:`sainsc.LazyKDE` object.
The data can be filtered, subset, and cropped to adjust the desired field of view and
genes.

In the next step the kernel for kernel density estimation (KDE) is defined and
cell types are assigned to each pixel using cell-type gene
expression signatures from e.g. single-cell RNAseq.

Otherwise you can find the local maxima of the KDE and treat these as proxies for cells.
From that point on you can proceed using standard single-cell RNAseq analysis and
spatial methods (e.g. using `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and
`squidpy <https://squidpy.readthedocs.io/en/stable/>`_).

Along the way you will want to (and should) generate a lot of plots to check your
results.

For a more concrete example of what a workflow looks like check out the
:doc:`usage` guide.
