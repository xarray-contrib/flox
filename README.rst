.. image:: https://img.shields.io/github/workflow/status/dcherian/dask_groupby/CI?logo=github&style=for-the-badge
    :target: https://github.com/dcherian/dask_groupby/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/github/workflow/status/dcherian/dask_groupby/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/dcherian/dask_groupby/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/dcherian/dask_groupby.svg?style=for-the-badge
    :target: https://codecov.io/gh/dcherian/dask_groupby

.. If you want the following badges to be visible, please remove this line, and unindent the lines below
    .. image:: https://img.shields.io/readthedocs/dask_groupby/latest.svg?style=for-the-badge
        :target: https://dask_groupby.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

    .. image:: https://img.shields.io/pypi/v/dask_groupby.svg?style=for-the-badge
        :target: https://pypi.org/project/dask_groupby
        :alt: Python Package Index

    .. image:: https://img.shields.io/conda/vn/conda-forge/dask_groupby.svg?style=for-the-badge
        :target: https://anaconda.org/conda-forge/dask_groupby
        :alt: Conda Version


dask_groupby
============

(See a `presentation <https://docs.google.com/presentation/d/1muj5Yzjw-zY8c6agjyNBd2JspfANadGSDvdd6nae4jg/edit?usp=sharing>`_ about this package).


This repo explores strategies for a distributed GroupBy with dask arrays. It was motivated by

1. Dask Dataframe GroupBy `blogpost <https://blog.dask.org/2019/10/08/df-groupby>`_
2. numpy_groupies in Xarray `issue <https://github.com/pydata/xarray/issues/4473>`_

The core GroupBy operation is outsourced to `numpy_groupies <https://github.com/ml31415/numpy-groupies>`_.
The GroupBy reduction is first applied blockwise. Those intermediate
results are combined by concatenating to form a new array which is then reduced
again. The combining of intermediate results uses dask's ``_tree_reduce`` till
all group results are in one block. At that point the result is "finalized" and
returned to the user. Here is an example of writing a custom Aggregation
(again inspired by dask.dataframe)


.. python::

    mean = Aggregation(
        # name used for dask tasks
        name="mean",
        # blockwise reduction
        chunk=("sum", "count"),
        # combine intermediate results: sum the sums, sum the counts
        combine=("sum", "sum"),
        # generate final result as sum / count
        finalize=lambda sum_, count: sum_ / count,
        # Used when "reindexing" at combine-time
        fill_value=0,
    )


The implementation with ``_tree_reduce`` complicates things.
An alternative simpler implementation would be to use the "tensordot"
`trick <https://github.com/dask/dask/blob/ac1bd05cfd40207d68f6eb8603178d7ac0ded922/dask/array/routines.py#L295-L310>`_.
But this requires knowledge of "expected group labels" at compute-time.

.. If you want the following badges to be visible, please remove this line, and unindent the lines below
    Re-create notebooks with Pangeo Binder
    --------------------------------------

    Try notebooks hosted in this repo on Pangeo Binder. Note that the session is ephemeral.
    Your home directory will not persist, so remember to download your notebooks if you
    made changes that you need to use at a later time!

    .. image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue&style=for-the-badge
        :target: https://binder.pangeo.io/v2/gh/dcherian/dask_groupby/master?urlpath=lab
        :alt: Binder
