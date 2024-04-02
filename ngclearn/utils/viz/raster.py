"""
Raster visualization functions/utilities.
"""
import math
import random
#import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import imageio.v3 as iio
import jax.numpy as jnp

#suffix = '.jpg' #.png

def _create_raster_plot(spike_train, ax=None, s=1.5, c="black", marker="|",
                       plot_fname=None, indices=None, suffix='.jpg'):
    spk_ = spike_train
    # Process spikes and create the binary coordinate grid
    if len(spk_.shape) == 1:
        spk_ = jnp.expand_dims(spk_,axis=1)
    n_units = spk_.shape[1]
    if indices is not None and indices.shape[0] < spk_.shape[1]:
        spk_ = spk_[:,indices] # access specific neurons if requested
        if len(spk_.shape) > 1:
            n_units = spk_.shape[1]
        else:
            n_units = spk_.shape[0]
    coords = spk_.nonzero() #jnp.where(spk_) #tf.where(spk_).numpy()
    spk_x = coords[0] #coords[:,0]
    spk_y = coords[1] #coords[:,1]
    if ax is not None:
        ax.scatter(spk_x, spk_y, s=s, c=c, marker=marker,linewidths=4)
        yint = range(0, n_units)
        ax.set_yticks(yint)
        ax.set_yticklabels(yint, fontsize=12)
        return ax
    else:
        if plot_fname is None:
            plot_fname = "raster_plot" + suffix
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.scatter(spk_x, spk_y, s=s, c=c, marker=marker)
        yint = range(0, n_units)
        ax.set_yticks(yint)
        ax.set_yticklabels(yint, fontsize=12)
        plt.title("Spike Train Raster Plot")
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Index")
        plt.savefig(plot_fname)
        plt.clf()
        plt.close()

def create_raster_plot(spike_train, ax=None, s=0.5, c="black",
                       plot_fname=None, indices=None, tag="", suffix='.jpg'):
    """
    Generates a raster plot of a given (binary) spike train (row dimension
    corresponds to the discrete time dimension).

    Args:
        spike_train: a numpy binary array of shape (T x number_neurons)

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
    n_count = spike_train.shape[0]
    step_count = spike_train.shape[1]
    save = False
    if ax is None:
        if plot_fname is None:
            plot_fname = "raster_plot" + suffix

        nc = n_count if indices is None else len(indices)
        fig_size = 5 if nc < 25 else int(nc / 5)
        plt.figure(figsize=(fig_size, fig_size)) # (fig_size * K, fig_size)
        plt.title("Spike Train Raster Plot, {}".format(tag))
        plt.xlabel("Time Step")
        # plt.ylabel("Neuron Index")
        save = True

    ax = ax if ax is not None else plt

    events = []
    for t in range(n_count):
        if indices is None or t in indices:
            e = spike_train[t,:].nonzero()
            events.append(e[0])
    ax.eventplot(events, linelengths=s, colors=c)
    ax.yticks(ticks=[i for i in (range(n_count if indices is None else len(indices)))],
              labels=["N" + str(i) for i in (range(n_count) if indices is None else indices)])
    ax.xticks(ticks=[i for i in range(0, step_count+1, max(int(step_count / 5), 1))])

    if save:
        ax.savefig(plot_fname)
        ax.clf()
        ax.close()
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
