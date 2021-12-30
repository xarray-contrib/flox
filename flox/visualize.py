import random
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import find_group_cohorts


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
    append=False,
):

    dx = 2
    xpts = x0 + np.arange(0, (ncol + nspaces) * dx, dx)
    ypts = np.arange(0, nrow * dx, dx)

    if colors is None:
        colors = mpl.cm.Set2.colors[:4]

    if not append:
        plt.figure()
        ax = plt.axes()
    else:
        ax = plt.gca()
    ax.set_aspect(1)
    ax.set_axis_off()

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
                (x, y - 0.5 * dx),
                dx,
                dx,
                edgecolor="w",
                linewidth=1,
                facecolor=fcolor,
            )
        )
        if draw_line_at is not None and icolor > 0 and icolor % draw_line_at == 0:
            plt.plot([x, x], [y - 0.75 * dx, y + 0.75 * dx], color="k", lw=2)

    ax.set_xlim((0, max(xpts) + dx))
    ax.set_ylim((-0.75 * dx, max(ypts) + 0.75 * dx))

    if not append:
        plt.gcf().set_size_inches((ncol * pxin, (nrow + 2) * pxin))


def visualize_groups_1d(array, labels, axis=-1, colors=None, cmap=None):
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

    plt.figure()
    i0 = 0
    for i in chunks:
        lab = labels[i0 : i0 + i]
        col = [colors[label] for label in lab] + [(1, 1, 1)]
        draw_mesh(
            1,
            len(lab) + 1,
            colors=col,
            randomize=False,
            append=True,
            x0=i0 * 2.3,  # + (i0 - 1) * 0.025,
        )
        i0 += i

    pxin = 0.8
    plt.gcf().set_size_inches((len(labels) * pxin, 1 * pxin))


def get_colormap(N):

    cmap = mpl.cm.get_cmap("tab20_r").copy()
    ncolors = len(cmap.colors)
    q = N // ncolors
    r = N % ncolors
    cmap = mpl.colors.ListedColormap(np.concatenate([cmap.colors] * q + [cmap.colors[:r]]))
    cmap.set_under(color="w")
    return cmap


def visualize_cohorts_2d(by, array, merge=True, method="cohorts"):
    assert by.ndim == 2
    print("finding cohorts...")
    cohorts = find_group_cohorts(
        by, [array.chunks[ax] for ax in range(-by.ndim, 0)], merge=merge, method=method
    )

    factorized = np.full(by.shape, -1)
    for idx, cohort in enumerate(cohorts):
        factorized[np.isin(by, cohort)] = idx
    ncohorts = idx

    xticks = np.cumsum(array.chunks[-1])
    yticks = np.cumsum(array.chunks[-2])

    f, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=True)

    flat = by.ravel()
    ngroups = len(np.unique(flat[~np.isnan(flat)]))

    h0 = ax[0].imshow(by, cmap=get_colormap(ngroups))
    h1 = ax[1].imshow(factorized, aspect="equal", vmin=0, cmap=get_colormap(ncohorts))
    for axx in ax:
        axx.grid(True, which="both")
        axx.set_xticks(xticks)
        axx.set_yticks(yticks)
    f.colorbar(h0, ax=ax[0])
    f.colorbar(h1, ax=ax[1])
    ax[0].set_title("by")
    ax[1].set_title("cohorts")
    f.set_size_inches((6, 6))
