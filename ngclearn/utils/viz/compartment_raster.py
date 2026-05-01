"""
Raster visualization functions/utilities.
"""
import matplotlib.pyplot as plt
import jax
from typing import Sequence

def create_raster_plot(spike_train: jax.Array, ax: plt.Axes | None = None,
                       indices: Sequence[int] | None = None, s=0.5, c="black"):
    """
    Generates a raster plot of a given (binary) spike train (row dimension
    corresponds to the discrete time dimension).

    Args:
        spike_train: a numpy binary array of shape (T x number_of_neurons)

        ax: a hook/pointer to a currently external plot that this raster plot
            should be made a sub-figure of

        indices: optional indices of neurons (row integer indices) to focus on
            plotting

        s: size of the spike scatter points (Default = 0.5)

        c: color of the spike scatter points (Default = black)

    """
    step_count = spike_train.shape[0]
    n_count = spike_train.shape[1]
    if ax is None:
        nc = n_count if indices is None else len(indices)
        fig_size = 5 if nc < 25 else int(nc / 5)
        plt.figure(figsize=(fig_size, fig_size))

    _ax = ax if ax is not None else plt

    events = []
    for t in range(n_count):
        if indices is None or t in indices:
            e = spike_train[:, t].nonzero()
            events.append(e[0])
    _ax.eventplot(events, linelengths=s, colors=c)
    if ax is None:
        _ax.yticks(ticks=[i for i in (range(n_count if indices is None else len(indices)))],
                  labels=["N" + str(i) for i in (range(n_count) if indices is None else indices)])
        _ax.xticks(ticks=[i for i in range(0, step_count+1, max(int(step_count / 5), 1))])
    else:
        _ax.set_yticks(ticks=[i for i in (range(n_count if indices is None else len(indices)))],
                  labels=["N" + str(i) for i in (range(n_count) if indices is None else indices)])
