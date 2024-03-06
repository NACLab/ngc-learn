"""
Visualization functions/utilities.
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
                       plot_fname=None, indices=None, tag="", suffix='.jpg'):
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
        plt.title("Spike Train Raster Plot, {}".format(tag))
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

def visualize(thetas, sizes, prefix, suffix='.jpg'):
    """

    Args:
        thetas:

        sizes:

        prefix:

        suffix:
    """
    Ts = [t.T for t in thetas] # [tf.transpose(t) for t in thetas]
    num_filters = [T.shape[0] for T in Ts]
    n_cols = [math.ceil(math.sqrt(nf)) for nf in num_filters]
    n_rows = [math.ceil(nf / c) for nf, c in zip(num_filters, n_cols)]

    starts = [sum(n_cols[:i]) + i for i in range(len(n_cols))]
    max_size = max(sizes)

    spacers = len(sizes) - 1
    n_cols_total = sum(n_cols) + spacers
    n_rows_total = max(n_rows)

    plt.figure(figsize=(n_cols_total, n_rows_total))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for idx in range(len(Ts)):
        T = Ts[idx]
        size = n_cols[idx]
        start = starts[idx]
        for i in range(num_filters[idx]):
            r = math.floor(i / n_cols[idx]) #math.sqrt(num_filters[idx]))
            extra = n_cols_total - size

            point = start + 1 + i + (r * extra)
            plt.subplot(n_rows_total, n_cols_total, point)
            filter = T[i, :]
            plt.imshow(np.reshape(filter, (sizes[idx][0], sizes[idx][1])), cmap=plt.cm.bone, interpolation='nearest')
            plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.savefig(prefix+suffix, bbox_inches='tight')
    plt.close()

def visualize_labels(thetas, sizes, prefix, space_width=None, widths=None, suffix='.jpg'):
    """

    Args:
        thetas:

        sizes:

        prefix:

        space_width:

        widths:

        suffix:
    """
    Ts = [t.T for t in thetas] # [tf.transpose(t) for t in thetas]
    num_filters = [T.shape[0] for T in Ts]
    n_cols = [math.ceil(math.sqrt(nf)) for nf in num_filters]
    n_rows = [math.ceil(nf / c) for nf, c in zip(num_filters, n_cols)]

    starts = [sum(n_cols[:i]) + i for i in range(len(n_cols))]

    spacers = len(sizes) - 1
    n_cols_total = sum(n_cols) + spacers
    n_rows_total = max(n_rows)

    max_height = max(sizes, key=lambda x: x[0])[0]
    max_width = max(sizes, key=lambda x: x[1])[1]
    fig = plt.figure()

    fig.set_figheight(max_height)
    fig.set_figwidth(max_width)

    if widths is None:
        _widths = [sizes[0][1] for _ in range(n_cols[0])]
        for i in range(1, len(n_cols)):
            _widths += [math.ceil(max_width / 2) if space_width is None else space_width] + [sizes[i][1] for _ in range(n_cols[i])]
    else:
        _widths = [widths[0] for _ in range(n_cols[0])]
        for i in range(1, len(n_cols)):
            _widths += [math.ceil(max(widths) / 2) if space_width is None else space_width] + [widths[i] for _ in range(n_cols[i])]


    spec = gridspec.GridSpec(ncols=n_cols_total, nrows=n_rows_total,
                             width_ratios=_widths, wspace=0.1,
                             hspace=0.1)

    # plt.figure(figsize=(n_cols_total*5, n_rows_total*2))
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for idx in range(len(Ts)):
        T = Ts[idx]
        size = n_cols[idx]
        start = starts[idx]
        for i in range(num_filters[idx]):
            r = math.floor(i / n_cols[idx]) #math.sqrt(num_filters[idx]))

            extra = n_cols_total - size

            point = start + i + (r * extra)
            # plt.subplot(n_rows_total, n_cols_total, point)
            ax = fig.add_subplot(spec[point])
            if r == 0:
                ax.set_title(str(chr(i + 65)), weight="bold")
            if i % n_cols[idx] == 0:
                ax.set_ylabel(str(r + 1), rotation=0, labelpad=15, size=max_height, weight="bold")
            filter = T[i, :]
            ax.imshow(np.reshape(filter, (sizes[idx][0], sizes[idx][1])), cmap=plt.cm.bone, interpolation='nearest')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])


    fig.subplots_adjust(top=0.9)
    fig.savefig(prefix+suffix, bbox_inches='tight')
    plt.close(fig)

def visualize_frame(frame, path='.', name='tmp', suffix='.jpg', **kwargs):
    iio.imwrite(path + '/' + name + suffix, frame.astype(jnp.uint8), **kwargs)

def visualize_gif(frames, path='.', name='tmp', suffix='.jpg', **kwargs):
    _frames = [f.astype(jnp.uint8) for f in frames]
    iio.imwrite(path + '/' + name + '.gif', _frames, **kwargs)

def make_video(f_start, f_end, path, prefix, suffix='.jpg', skip=1):
    images = []
    for i in range(f_start, f_end+1, skip):
        print("Reading frame " + str(i))
        images.append(iio.imread(path + "/" + prefix + str(i) + suffix))
    print("writing gif")
    iio.imwrite(path + '/training.gif', images, loop=0, duration=200)

#if __name__ == "__main__":
#    make_video(0, 50, 'trials/Two_Layer/3', 'joint_')
