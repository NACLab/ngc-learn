"""
Synaptic/receptive field visualization functions/utilities.
"""
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import imageio.v3 as iio
import jax.numpy as jnp



def visualize(thetas, sizes, prefix, order=None, suffix='.jpg'):
    """

    Args:
        thetas:

        sizes:

        prefix:

        suffix:
    """


    if order is None:
        order = ['C' for _ in range(len(thetas))]

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
            plt.imshow(np.reshape(filter, (sizes[idx][0], sizes[idx][1]), order=order[idx]), cmap=plt.cm.bone, interpolation='nearest')
            plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.savefig(prefix+suffix, bbox_inches='tight')
    plt.clf()
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

def make_video(f_start, f_end, path, prefix, suffix='.jpg', skip=1, **kwargs):
    images = []
    for i in range(f_start, f_end+1, skip):
        print("Reading frame " + str(i))
        images.append(iio.imread(path + "/" + prefix + str(i) + suffix))
    print("writing gif")
    iio.imwrite(path + '/training.gif', images, **kwargs)


# def visualize_norm(thetas, sizes, prefix, suffix='.jpg'):

def viz_block(thetas, sizes, prefix, suffix=".jpg", padding=1, low_rez=True):
    num_filters = [T.shape[1] for T in thetas]
    n_cols = [math.ceil(math.sqrt(nf)) for nf in num_filters]
    n_rows = [math.ceil(nf / c) for nf, c in zip(num_filters, n_cols)]
    idxs = [i for i in range(len(thetas))]

    if not low_rez:
        n_cols_size = int(sum([(t_c * cols) + padding * (cols - 1) for (t_c, _), cols in zip(sizes, n_cols)]))
        n_rows_size = int(sum([(t_r * rows) + padding * (rows - 1) for (_, t_r), rows in zip(sizes, n_rows)]))
        plt.figure(figsize=(n_cols_size, n_rows_size))


    for t, num_f, (t_c, t_r), cols, rows, idx in zip(thetas, num_filters, sizes, n_cols, n_rows, idxs):
        c_dim = (t_c * cols) + padding * (cols - 1)
        r_dim = (t_r * rows) + padding * (rows - 1)

        full = jnp.ones((r_dim, c_dim)) * np.amax(t)

        for k in range(num_f):
            r = k // cols
            c = k % cols

            r_start = (r * (t_r + padding))
            r_end = (r * (t_r + padding)) + t_r

            c_start = (c * (t_c + padding))
            c_end = (c * (t_c + padding)) + t_c

            full = full.at[r_start:r_end, c_start:c_end].set(
                jnp.reshape(t[:, k], (t_r, t_c)))

        plt.subplot(1, len(thetas), idx+1)
        plt.imshow(full, cmap=plt.cm.bone, interpolation='nearest')
        plt.axis("off")

    plt.savefig(prefix + suffix, bbox_inches='tight')
    plt.clf()
    plt.close()

