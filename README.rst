.. image:: https://img.shields.io/github/workflow/status/xarray-contrib/dask_groupby/CI?logo=github&style=for-the-badge
    :target: https://github.com/xarray-contrib/dask_groupby/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/github/workflow/status/xarray-contrib/dask_groupby/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/xarray-contrib/dask_groupby/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/xarray-contrib/dask_groupby.svg?style=for-the-badge
    :target: https://codecov.io/gh/xarray-contrib/dask_groupby

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

Development
------------

For a development install, do the following in the repository directory:

.. code-block:: bash

    conda env update -f ci/environment.yml
    conda activate sandbox-devel
    python -m pip install -e .

Also, please install `pre-commit` hooks from the root directory of the created project by running::

      pre-commit install

These code style pre-commit hooks (black, isort, flake8, ...) will run every time you are about to commit code.

.. If you want the following badges to be visible, please remove this line, and unindent the lines below
    Re-create notebooks with Pangeo Binder
    --------------------------------------

    Try notebooks hosted in this repo on Pangeo Binder. Note that the session is ephemeral.
    Your home directory will not persist, so remember to download your notebooks if you
    made changes that you need to use at a later time!

    .. image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue&style=for-the-badge
        :target: https://binder.pangeo.io/v2/gh/xarray-contrib/dask_groupby/master?urlpath=lab
        :alt: Binder
