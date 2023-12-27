import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from bpp.qubo import Qubo


def visu_mat(matrix: ArrayLike, n_colors: int = 10) -> None:
    """
    Visualize a matrix with a colorbar. Zero values remain uncolored.

    Parameters
    ----------
    matrix : ArrayLike
        The matrix to visualize.
    n_colors : int, optional
        The number of colors to use in the colorbar, by default 10.
    """
    matrix = np.array(matrix)

    value_range = 0, 1
    if np.any(matrix != 0):
        value_range = np.floor(np.min(matrix[matrix != 0])), np.ceil(np.max(matrix))

    fig, ax = plt.subplots()

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=value_range[0], vmax=value_range[1])
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    cax = ax.imshow(masked_matrix, cmap=cmap, norm=norm)

    cbar_ticks = np.linspace(value_range[0], value_range[1], n_colors)
    fig.colorbar(cax, ax=ax, ticks=cbar_ticks)

    lims = matrix.shape
    xlim = (-0.5, lims[1] - 0.5)
    ylim = (-0.5, lims[0] - 0.5)
    for x in range(int(xlim[1])):
        ax.vlines(x + 0.5, ymin=ylim[0], ymax=ylim[1], color="grey", lw=0.2, ls=":")
    for y in range(int(ylim[1])):
        ax.hlines(y + 0.5, xmin=xlim[0], xmax=xlim[1], color="grey", lw=0.2, ls=":")

    plt.show()


def plot_bins(solution_vector: ArrayLike, qubo: Qubo):
    n_items = qubo.n_items
    n_bins = qubo.n_bins
    bins = qubo.bin_capacities
    items = qubo.weights

    mat = np.array(solution_vector)[: n_items * n_bins]
    mat = mat.reshape((n_items, n_bins))

    fig, ax = plt.subplots()
    for bin in range(n_bins):
        capacity = bins[bin]
        ax.bar(
            str(bin),
            capacity,
            bottom=0,
            color="lightgray",
            edgecolor="black",
            linewidth=1,
            alpha=0.5,
        )

        mask = mat[:, bin] == 1
        idx_lst = np.array(range(n_items))[mask]
        w0 = 0

        for idx in idx_lst:
            weight = items[idx]

            ax.bar(str(bin), weight, bottom=w0, alpha=0.5)
            w0 += weight

            ax.text(
                str(bin),
                w0 - weight / 2,
                str(idx),
                ha="center",
                va="center",
                color="white",
            )
