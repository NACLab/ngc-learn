"""
Visualization functions/utilities.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

def plot_learning_curves(acc_curve, dev_acc_curve, plot_fname=None, scale=1.0,
                         y_lab="Loss", x_lab="Iteration"):
    """

    Args:
        acc_curve: a numpy array of shape (T x 1), where T is number of
            epochs/iterations (array of training scalar measurements)

        dev_acc_curve: a numpy array of shape (T x 1), where T is number of
            epochs/iterations (array of validation/test scalar measurements)

        fname: the filename to save this plot as, i.e., /path/to/name.png
            (Default: lif_analysis.png)

        scale: multiplier to scale curves by (e.g., scale = 100 for
            accuracy/error measurements)
    """
    colors = ["red", "blue"]
    # plot loss learning curves
    y = acc_curve
    vy = dev_acc_curve
    y = y * scale
    vy = vy * scale
    x_iter = np.asarray(list(range(0, y.shape[0])))
    fontSize = 18 #20
    plt.plot(x_iter, y, '-', color=colors[0])
    plt.plot(x_iter, vy, '-', color=colors[1])
    plt.xlabel(x_lab, fontsize=fontSize)
    plt.ylabel(y_lab, fontsize=fontSize)
    plt.grid()
    acc = mpatches.Patch(color=colors[0], label='Train')
    vacc = mpatches.Patch(color=colors[1], label='Dev')
    plt.legend(handles=[acc, vacc], fontsize=13, ncol=2,borderaxespad=0, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, -0.175))#, prop=fontP)
    plt.tight_layout()
    if plot_fname is None:
        plot_fname = "learning_curves.jpg"
    plt.savefig(plot_fname)
    plt.clf()

def create_raster_plot(spike_train, ax=None, s=1.5, c="black", marker="|",
                       plot_fname=None, indices=None):
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
    """
    spk_ = spike_train
    # Process spikes and create the binary coordinate grid
    if len(spk_.shape) == 1:
        spk_ = np.expand_dims(spk_,axis=1)
    n_units = spk_.shape[1]
    if indices is not None and indices.shape[0] < spk_.shape[1]:
        spk_ = spk_[:,indices] # access specific neurons if requested
        if len(spk_.shape) > 1:
            n_units = spk_.shape[1]
        else:
            n_units = spk_.shape[0]
    coords = tf.where(spk_).numpy()
    spk_x = coords[:,0]
    spk_y = coords[:,1]
    if ax is not None:
        ax.scatter(spk_x, spk_y, s=s, c=c, marker=marker,linewidths=4)
        yint = range(0, n_units)
        ax.set_yticks(yint)
        ax.set_yticklabels(yint, fontsize=12)
        #ax.set_ylim([-((s*1.0)/2.0), ((s*1.0)/2.0)])
        #ax.set_ylim([-20, 20])
        return ax
    else:
        if plot_fname is None:
            plot_fname = "raster_plot.png"
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

def plot_spiking_neuron(curr, mem, spike, ref, dt, thr_line=False,
                    title=False, min_mem_val=0.0, max_mem_val=1.25, fname=None):
    """
    Simple plotting function for visualizing the trajectory of a single neuron
    (where its input electrical current, membrane potential value, output
    spike readings, and refractory variable states have been recorded into arrays).

    Args:
        curr: the recorded electrical current (over T steps)

        mem: the recorded membrane potential (over T steps)

        spike: the recorded spike train (over T steps)

        ref: the recorded refractory state value (over T steps)

        dt: the integration time constant

        thr_line: optional vertical threshold line to plot for voltage

        title: the title of the plot

        min_mem_val: minimum value bound on membrane potential subplot (min of y-axis)

        max_mem_val: maximum value bound on membrane potential subplot (max of y-axis)

        fname: the filename to save this plot as, i.e., /path/to/name.png (Default: lif_analysis.png)
    """
    x_lim = curr.shape[0]
    y_curr_lim = float(np.amax(curr)) + 0.2
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})
    # plot input current
    ax[0].plot(curr, c="tab:blue")
    ax[0].set_ylim([0, y_curr_lim])
    ax[0].set_xlim([0, x_lim])
    ax[0].set_ylabel("Input Current ($J_t$)")
    if title:
        ax[0].set_title(title)

    # plot membrane potential
    ax[1].plot(mem, c="tab:red")
    ax[1].set_ylim([min_mem_val, max_mem_val])
    ax[1].set_ylabel("Membrane Potential ($V_t$)")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time Step $t$")

    spk_ = spike + 0
    mask = np.greater_equal(ref,dt).astype(np.float32)
    spk_ = spk_ * (1.0 - mask)

    # Plot output spike using spikeplot
    create_raster_plot(spk_, ax[2], s=100, c="black")
    plt.ylabel("Output Spikes")
    plt.yticks([])
    if fname is None:
        fname = "lif_plot.png"
    plt.savefig(fname)
    plt.clf()
