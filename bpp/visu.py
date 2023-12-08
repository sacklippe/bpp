import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


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
