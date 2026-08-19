"""
Synaptic/receptive field visualization functions/utilities.
"""
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import imageio.v3 as iio
import jax.numpy as jnp


def visualize_macro_grid( ## more complex filter visualization co-routine
        thetas,
        sizes,
        macro_grid_shape,
        prefix,
        order=None,
        suffix='.jpg',
        contrast_by_data=True
):
    """
    Stitches filter sets into a 2D Grid-of-Grids layout with bright white padding.

    Args:
        thetas:
        sizes:
        macro_grid_shape:
        prefix:
        order:
        suffix:
        contrast_by_data:

    Returns:

    """
    if order is None:
        order = ['C' for _ in range(len(thetas))]

    H_macro, W_macro = macro_grid_shape
    num_banks = len(thetas)

    T_sample = thetas[0].T
    filters_per_bank = T_sample.shape[0]
    f_cols = int(math.ceil(math.sqrt(filters_per_bank)))
    f_rows = int(math.ceil(filters_per_bank / f_cols))

    p_h, p_w = sizes[0]
    ## total pixel height/width of an individual composite filter bank block
    bank_px_h = f_rows * p_h
    bank_px_w = f_cols * p_w

    ## Use stark white background; initialize the entire master sheet canvas with 1.0 (White in grayscale)
    pad = 2
    canvas_h = H_macro * bank_px_h + (H_macro + 1) * pad
    canvas_w = W_macro * bank_px_w + (W_macro + 1) * pad
    master_canvas = np.ones((canvas_h, canvas_w))

    for b_idx in range(num_banks):
        m_row = b_idx // W_macro
        m_col = b_idx % W_macro
        if m_row >= H_macro:
            break

        T = thetas[b_idx].T
        b_start_y = m_row * bank_px_h + (m_row + 1) * pad
        b_start_x = m_col * bank_px_w + (m_col + 1) * pad
        for f_idx in range(filters_per_bank):
            if f_idx >= T.shape[0]:
                break
            i_row = f_idx // f_cols
            i_col = f_idx % f_cols
            single_filter = np.reshape(T[f_idx, :], (p_h, p_w), order=order[b_idx])

            ## shift values to map correctly to the bone color scheme; max absolute value normalization
            max_val = float(np.max(np.abs(single_filter)))
            if max_val > 0:
                single_filter = single_filter / max_val

            y_loc = b_start_y + (i_row * p_h)
            x_loc = b_start_x + (i_col * p_w)
            master_canvas[y_loc:y_loc + p_h, x_loc:x_loc + p_w] = single_filter

    ## render out the crisp grid sheet
    plt.figure(figsize=(10, 10), dpi=300)

    ## use vmin=-1.0 and vmax=1.0 so that the 1.0 canvas background registers as absolute white
    max_val = None #1.
    min_val = None #-1.
    if contrast_by_data:
        max_val = float(jnp.max(jnp.abs(master_canvas)))
        min_val = float(jnp.min(jnp.abs(master_canvas)))
    plt.imshow(master_canvas, cmap=plt.cm.bone, interpolation='nearest', vmin=min_val, vmax=max_val)
    plt.axis("off")
    plt.savefig(prefix + suffix, bbox_inches='tight', pad_inches=0.0)
    plt.clf()
    plt.close()


def visualize(
        thetas,
        sizes,
        prefix,
        order=None,
        suffix='.jpg'
):
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
            _filter = T[i, :]
            max_val = float(jnp.max(jnp.abs(_filter)))
            min_val = float(jnp.min(jnp.abs(_filter)))
            plt.imshow(
                np.reshape(_filter, (sizes[idx][0], sizes[idx][1]), order=order[idx]), 
                cmap=plt.cm.bone, interpolation='nearest', vmin=min_val, vmax=max_val
            )
            plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.savefig(prefix+suffix, bbox_inches='tight')
    plt.clf()
    plt.close()


def visualize_labels(
        thetas,
        sizes,
        prefix,
        space_width=None,
        widths=None,
        suffix='.jpg'
):
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

def visualize_frame(
        frame,
        path='.',
        name='tmp',
        suffix='.jpg',
        **kwargs
):
    iio.imwrite(path + '/' + name + suffix, frame.astype(jnp.uint8), **kwargs)

def visualize_gif(
        frames,
        path='.',
        name='tmp',
        suffix='.jpg',
        **kwargs
):
    _frames = [f.astype(jnp.uint8) for f in frames]
    iio.imwrite(path + '/' + name + '.gif', _frames, **kwargs)

def make_video(
        f_start,
        f_end,
        path,
        prefix,
        suffix='.jpg',
        skip=1,
        **kwargs
):
    images = []
    for i in range(f_start, f_end+1, skip):
        print("Reading frame " + str(i))
        images.append(iio.imread(path + "/" + prefix + str(i) + suffix))
    print("writing gif")
    iio.imwrite(path + '/training.gif', images, **kwargs)

def viz_block(
        thetas,
        sizes, prefix,
        suffix=".jpg",
        padding=1,
        low_rez=True
):
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

