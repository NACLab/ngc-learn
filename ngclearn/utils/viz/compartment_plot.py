"""
Raster visualization functions/utilities.
"""
import matplotlib.pyplot as plt
import jax
from typing import Sequence

def create_plot(history: jax.Array, ax: plt.Axes | None = None,
                       indices: Sequence[int] | None = None):
    """
    Generates a raster plot of a given (binary) spike train (row dimension
    corresponds to the discrete time dimension).

    Args:
        history: a numpy binary array of shape (T x number_of_neurons)

        ax: a hook/pointer to a currently external plot that this raster plot
            should be made a sub-figure of

        indices: optional indices of neurons (row integer indices) to focus on
            plotting

        s: size of the spike scatter points (Default = 0.5)

        c: color of the spike scatter points (Default = black)

    """
    n_count = history.shape[0]
    if ax is None:
        nc = n_count if indices is None else len(indices)
        fig_size = 5 if nc < 25 else int(nc / 5)
        plt.figure(figsize=(fig_size, fig_size))

    _ax = ax if ax is not None else plt

    for k in range(history.shape[1]):
        if indices is None or k in indices:
            _ax.plot(history[:, k])