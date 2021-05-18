#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

with open("README.md") as f:
    long_description = f.read()

setup(
    maintainer="Deepak Cherian",
    maintainer_email="deepak@cherian.net",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    description="GroupBy operations for dask.array",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=long_description,
    include_package_data=True,
    keywords="dask_groupby",
    name="dask_groupby",
    packages=find_packages(include=["dask_groupby", "dask_groupby.*"]),
    url="https://github.com/dcherian/dask_groupby",
    project_urls={
        "Documentation": "https://github.com/dcherian/dask_groupby",
        "Source": "https://github.com/dcherian/dask_groupby",
        "Tracker": "https://github.com/dcherian/dask_groupby/issues",
    },
    zip_safe=False,
)
