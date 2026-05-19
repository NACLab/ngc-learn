import matplotlib.pyplot as plt
import numpy as np

def visualize_confusion_heatmap(
    confuse_matrix, 
    classes, 
    out_fname, 
    figure_title="Confusion Matrix", 
    color_map="Blues", # "Greens" "Reds"
    fontsize=10, 
    show_percent=True
):
    _conf = confuse_matrix
    pscale = 1. ## percentage scale
    if show_percent:
        _conf = (_conf / np.sum(_conf, axis=1, keepdims=True)) * 100
    ## Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = np.floor(np.min(_conf))
    vmax = np.ceil(np.max(_conf))
    im = ax.imshow(
        _conf, interpolation="nearest", cmap=color_map, vmin=vmin, vmax=vmax
    )
    ## Add color bar
    fig.colorbar(im, ax=ax, shrink=0.75)
    ## Configure axes labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(figure_title, fontsize=14, pad=15)

    ## Loop to print numbers on top of each colored cell
    threshold = _conf.max() / 2.0  # Find midpoint to flip text color for readability
    for i in range(_conf.shape[0]):
        for j in range(_conf.shape[1]):
            ## Use white text on dark cells, black text on light cells
            color = "white" if _conf[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{_conf[i, j]:.1f}", #format(confuse_matrix[i, j], "d"),
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                weight="bold",
            )

    ## Ensure layout fits tightly
    plt.tight_layout()
    plt.savefig(out_fname)

