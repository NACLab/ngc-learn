"""
Visualization functions/utilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def create_raster_plot(spike_train, ax=None, s=1.5, c="black", marker="|",
                       plot_fname=None):
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
    """
    spk_ = spike_train
    # Process spikes and create the binary coordinate grid
    if len(spk_.shape) == 1:
        spk_ = np.expand_dims(spk_,axis=1)
    coords = tf.where(spk_).numpy()
    spk_x = coords[:,0]
    spk_y = coords[:,1]
    if ax is not None:
        return ax.scatter(spk_x, spk_y, s=s, c=c, marker=marker)
    else:
        if plot_fname is None:
            plot_fname = "raster_plot.png"
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.scatter(spk_x, spk_y, s=s, c=c, marker=marker)
        plt.title("Spike Train Raster Plot")
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Index")
        plt.savefig(plot_fname)
