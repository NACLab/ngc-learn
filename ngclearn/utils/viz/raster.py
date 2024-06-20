"""
Raster visualization functions/utilities.
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import imageio.v3 as iio
import jax.numpy as jnp

def create_raster_plot(spike_train, ax=None, s=0.5, c="black",
                       plot_fname=None, indices=None, tag="", suffix='.jpg', 
                       title_font_size=20):
    """
    Generates a raster plot of a given (binary) spike train (row dimension
    corresponds to the discrete time dimension).

    Args:
        spike_train: a numpy binary array of shape (number_neurons x 1 x T) OR
            shape (number_neurons x T)

        ax: a hook/pointer to a currently external plot that this raster plot
            should be made a sub-figure of

        s: size of the spike scatter points (Default = 1.5)

        c: color of the spike scatter points (Default = black)

        plot_fname: if ax is None, then this is the file name of the raster plot
            saved to disk (if plot_fname and ax are both None, then default
            plot_fname will be "raster_plot.png" and saved locally)

        indices: optional indices of neurons (row integer indices) to focus on plotting

        tag:

        suffix: output plot file suffix name to append
    """
    if len(spike_train.shape) == 3 and spike_train.shape[1] == 1:
        spike_train = jnp.transpose(spike_train, [2,0,1])
    elif len(spike_train.shape) == 2:
        spike_train = spike_train.T

    n_count = spike_train.shape[0]
    step_count = spike_train.shape[1]
    save = False
    if ax is None:
        if plot_fname is None:
            plot_fname = "raster_plot" + suffix

        nc = n_count if indices is None else len(indices)
        fig_size = 5 if nc < 25 else int(nc / 5)
        plt.figure(figsize=(fig_size, fig_size)) # (fig_size * K, fig_size)
        plt.title("Spike Train Raster Plot, {}".format(tag), fontsize=title_font_size)
        plt.xlabel("Time Step")
        # plt.ylabel("Neuron Index")
        save = True

    _ax = ax if ax is not None else plt

    events = []
    for t in range(n_count):
        if indices is None or t in indices:
            e = spike_train[t, :].nonzero()
            events.append(e[0])
    _ax.eventplot(events, linelengths=s, colors=c)
    if ax is None:
        _ax.yticks(ticks=[i for i in (range(n_count if indices is None else len(indices)))],
                  labels=["N" + str(i) for i in (range(n_count) if indices is None else indices)])
        _ax.xticks(ticks=[i for i in range(0, step_count+1, max(int(step_count / 5), 1))])

    if save:
        _ax.savefig(plot_fname)
        _ax.clf()
        _ax.close()
        plt.close()

def create_overlay_raster_plot(spike_train, targ_train, Y, idxs, s=1.5, c="black", marker="|",
                               plot_fname=None, indices=None, end_time=100, delay=10,
                               suffix='.jpg'):
    """
    Generates a raster plot of a given (binary) spike train (row dimension
    corresponds to the discrete time dimension).

    Args:
        spike_train: a numpy binary array of shape (T x number_neurons)

        ax: a hook/pointer to a currently external plot that this raster plot
            should be made a sub-figure of

        s: size of the spike scatter points (Default = 1.5)

        c: color of the spike scatter points (Default = black)

        marker: format of the marker used to represent each spike (Default = "|")

        plot_fname: if ax is None, then this is the file name of the raster plot
            saved to disk (if plot_fname and ax are both None, then default
            plot_fname will be "raster_plot.png" and saved locally)

        indices: optional indices of neurons (row integer indices) to focus on plotting

        end_time:

        delay:

        suffix: output plot file suffix name to append
    """
    for idx in idxs:
        spk_ = jnp.concatenate([jnp.expand_dims(s[idx, :], axis=0) for s in spike_train], axis=0)
        trg_ = jnp.concatenate([jnp.expand_dims(s[idx, :], axis=0) for s in targ_train], axis=0)
        tag = "Label: " + str(jnp.argmax(Y[idx,:]))
        n_units = spk_.shape[1]

        correct_spikes = jnp.where(spk_ + trg_ == 2.) # black
        failed_spikes = jnp.where(spk_ < trg_) # blue
        extra_spikes = jnp.where(spk_ > trg_) # red

        if plot_fname is None:
            plot_fname = "raster_plot" + suffix
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)

        _x = correct_spikes[0] #coords[:,0]
        _y = correct_spikes[1] #coords[:,1]
        ax.scatter(_x, _y, s=s, c="black", marker=marker)
        _x = failed_spikes[0] #coords[:,0]
        _y = failed_spikes[1] #coords[:,1]
        ax.scatter(_x, _y, s=s, c="blue", marker=marker)
        _x = extra_spikes[0] #coords[:,0]
        _y = extra_spikes[1] #coords[:,1]
        ax.scatter(_x, _y, s=s, c="red", marker=marker)

        yint = range(0, n_units)
        ax.set_yticks(yint)
        ax.set_yticklabels(yint, fontsize=12)
        ax.xaxis.set_ticks(jnp.arange(0, end_time + delay, delay))
        plt.title("Overlay Raster Plot, {}".format(tag))
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Index")
        plt.savefig(plot_fname + '_' + str(idx) + suffix)
        plt.clf()
        plt.close()
