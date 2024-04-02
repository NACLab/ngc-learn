import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet
import jax.numpy as jnp
import numpy as np
from ngclearn.utils.viz.raster import _create_raster_plot

def plot_spiking_neuron(curr, mem, spike, ref, dt, thr_line=False,
                    title=False, min_mem_val=0.0, max_mem_val=1.25, fname=None):
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
    mask = 0.
    if ref != None:
        mask = np.greater_equal(ref,dt).astype(np.float32)
    spk_ = spk_ * (1.0 - mask)

    # Plot output spike using spikeplot
    _create_raster_plot(spk_, ax[2], s=100, c="black")
    plt.ylabel("Output Spikes")
    plt.yticks([])
    if fname is None:
        fname = "lif_plot.png"
    plt.savefig(fname)
    plt.clf()
