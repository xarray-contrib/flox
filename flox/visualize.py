import random
from itertools import product

import numpy as np
import pandas as pd


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

    import matplotlib as mpl
    import matplotlib.pyplot as plt

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


def visualize_groups(array, labels, axis=-1, colors=None, cmap=None):
    """
    Visualize group distribution for a 1D array of group labels.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
