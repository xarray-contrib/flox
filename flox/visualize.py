import random
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import _unique, find_group_cohorts


def draw_mesh(
    nrow,
    ncol,
    *,
    draw_line_at=None,
    nspaces=0,
    space_at=0,
    pxin=0.3,
    counter=None,
    colors=None,
    randomize=True,
    x0=0,
    y0=0,
    append=False,
):
    dx = 2
    xpts = x0 + np.arange(0, (ncol + nspaces) * dx, dx)
    ypts = y0 + np.arange(0, nrow * dx, dx)

    if colors is None:
        colors = mpl.cm.Set2.colors[:4]

    if not append:
        plt.figure()
        ax = plt.axes()
    else:
        ax = plt.gca()
    ax.set_aspect(1)
    ax.set_axis_off()

    # ncolors = len(colors)
    if not randomize:
        colors = iter(colors)

    icolor = -1
    for n, (y, x) in enumerate(product(ypts, xpts)):
        if space_at > 0 and (n % space_at) == 0:
            continue
        if randomize:
            fcolor = random.choice(colors)
        else:
            fcolor = next(colors)
            icolor += 1
        if counter is not None:
            counter[fcolor] += 1
        ax.add_patch(
            mpl.patches.Rectangle(
                (x, y),
                dx,
                dx,
                edgecolor="w",
                linewidth=1,
                facecolor=fcolor,
            )
        )
        if draw_line_at is not None and icolor > 0 and icolor % draw_line_at == 0:
            plt.plot([x, x], [y - 0.75 * dx, y + 0.75 * dx], color="k", lw=2)

    # assert n + 1 == ncolors, (n, ncolors)
    ax.set_xlim((0, max(xpts) + 2 * dx))
    ax.set_ylim((-0.75 * dx + min(ypts), max(ypts) + 0.75 * dx))

    if not append:
        plt.gcf().set_size_inches((ncol * pxin, (nrow + 2) * pxin))


def visualize_groups_1d(array, labels, axis=-1, colors=None, cmap=None, append=True, x0=0):
    """
    Visualize group distribution for a 1D array of group labels.
    """

    labels = np.asarray(labels)
    assert labels.ndim == 1
    factorized, unique_labels = pd.factorize(labels)
    assert np.array(labels).ndim == 1
    chunks = array.chunks[axis]

    if colors is None:
        if cmap is None:
            colors = list(mpl.cm.tab20.colors)
        elif cmap is not None:
            colors = [cmap((num - 1) / len(unique_labels)) for num in unique_labels]

    if len(unique_labels) > len(colors):
        raise ValueError("Not enough unique colors")

    if not append:
        fig = plt.figure()
    i0 = 0
    for i in chunks:
        lab = labels[i0 : i0 + i]
        col = [colors[label] for label in lab] + [(1, 1, 1)]
        draw_mesh(
            1,
            len(lab) + 1,
            colors=col,
            randomize=False,
            append=append,
            x0=x0 + i0 * 2.3,  # + (i0 - 1) * 0.025,
        )
        i0 += i

    if not append:
        pxin = 0.8
        fig.set_size_inches((len(labels) * pxin, 1 * pxin))


def get_colormap(N):
    cmap = mpl.cm.get_cmap("tab20_r").copy()
    ncolors = len(cmap.colors)
    q = N // ncolors
    r = N % ncolors
    cmap = mpl.colors.ListedColormap(np.concatenate([cmap.colors] * q + [cmap.colors[: r + 1]]))
    cmap.set_under(color="k")
    return cmap


def factorize_cohorts(chunks, cohorts):
    chunk_grid = tuple(len(c) for c in chunks)
    nchunks = np.prod(chunk_grid)
    factorized = np.full((nchunks,), -1, dtype=np.int64)
    for idx, cohort in enumerate(cohorts):
        factorized[list(cohort)] = idx
    return factorized.reshape(chunk_grid)


def visualize_cohorts_2d(by, chunks):
    assert by.ndim == 2
    print("finding cohorts...")
    chunks = [chunks[ax] for ax in range(-by.ndim, 0)]
    _, chunks_cohorts = find_group_cohorts(by, chunks)
    print("finished cohorts...")

    xticks = np.cumsum(chunks[-1])
    yticks = np.cumsum(chunks[-2])

    f, ax = plt.subplots(1, 2, constrained_layout=True, sharex=False, sharey=False)
    ax = ax.ravel()
    # ax[1].set_visible(False)
    # ax = ax[[0, 2, 3]]

    ngroups = len(_unique(by))
    h0 = ax[0].imshow(by, vmin=0, cmap=get_colormap(ngroups))
    h2 = _visualize_cohorts(chunks, chunks_cohorts, ax=ax[1])

    ax[0].grid(True, which="both")
    for axx in ax[:1]:
        axx.set_xticks(xticks)
        axx.set_yticks(yticks)
    for h, axx in zip([h0, h2], ax):
        f.colorbar(h, ax=axx, orientation="horizontal")

    ax[0].set_title(f"by: {ngroups} groups")
    ax[1].set_title(f"{len(chunks_cohorts)} cohorts")
    f.set_size_inches((9, 6))


def _visualize_cohorts(chunks, cohorts, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    data = factorize_cohorts(chunks, cohorts)
    return ax.imshow(data, vmin=0, cmap=get_colormap(len(cohorts)))


def visualize_groups_2d(labels, y0=0, **kwargs):
    colors = mpl.cm.tab10_r
    for _i, chunk in enumerate(labels):
        chunk = np.atleast_2d(chunk)
        draw_mesh(
            *chunk.shape,
            colors=tuple(colors(label) for label in np.flipud(chunk).ravel()),
            randomize=False,
            append=True,
            y0=y0,
            **kwargs,
        )
        y0 = y0 + 2 * chunk.shape[0] + 2
    plt.ylim([-1, y0])
