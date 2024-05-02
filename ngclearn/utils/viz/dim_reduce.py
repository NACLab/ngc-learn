import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

def extract_pca_latents(vectors): ## PCA mapping routine
    """
    Projects collection of K vectors (stored in a matrix) to a two-dimensional (2D)
    visualization space via principal components analysis (PCA). Note that
    if the input already has a 2D dimensionality, the original input is returned.

    Args:
        vectors: a matrix/codebook of (K x D) vectors to project

    Returns:
        a matrix (K x 2) of projected vectors (to 2D space)
    """
    batch_size = 50
    z_dim = vectors.shape[1]
    z_2D = None
    if z_dim != 2:
        ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
        ipca.fit(vectors)
        z_2D = ipca.transform(vectors)
    else:
        z_2D = vectors
    return z_2D

def extract_tsne_latents(vectors, perplexity=30, n_pca_comp=32): ## tSNE mapping routine
    """
    Projects collection of K vectors (stored in a matrix) to a two-dimensional (2D)
    visualization space via the t-distributed stochastic neighbor embedding
    algorithm (t-SNE). This algorithm also uses PCA to produce an
    intermediate project to speed up the t-SNE final mapping step. Note that
    if the input already has a 2D dimensionality, the original input is returned.

    Args:
        vectors: a matrix/codebook of (K x D) vectors to project

        perplexity: the perplexity control factor for t-SNE (Default: 30)

    Returns:
        a matrix (K x 2) of projected vectors (to 2D space)
    """
    batch_size = 500 #50
    z_dim = vectors.shape[1]
    z_2D = None
    if z_dim != 2:
        print(" > Projecting latents via iPCA...")
        n_comp = n_pca_comp #32 #10 #16 #50
        if n_comp > batch_size:
            batch_size = n_comp
        if vectors.shape[1] < n_comp:
            n_comp = vectors.shape[1] - 2 #z_top.shape[1]-2
            n_comp = max(2, n_comp)
        ipca = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
        ipca.fit(vectors)
        z_2D = ipca.transform(vectors)
        print(" PCA.lat.shape = ",z_2D.shape)
        print(" > Finishing projection via t-SNE...")
        z_2D = TSNE(n_components=2,perplexity=perplexity, verbose=1).fit_transform(z_2D)
    else:
        z_2D = vectors
    return z_2D

def plot_latents(code_vectors, labels, plot_fname="2Dcode_plot.jpg", alpha=1.):
    """
    Produces a label-overlaid (label map to distinct colors) scatterplot for
    visualizing two-dimensional latent codes (produced by either PCA or t-SNE).

    Args:
        code_vectors: a matrix of shape (K x 2) with vectors to plot/visualize

        labels: label values, either of shape (K x 1) of integer values or of
            shape (K x C) of binary one-hot encodings where C is the number of
            classes.

        plot_fname: /path/to/plot_fname.<suffix> for saving the plot to disk
    """
    print(" > Plotting 2D latent encodings...")
    lab = labels
    if lab.shape[1] > 1: ## extract integer class labels from a one-hot matrix
        lab = np.argmax(lab, 1)
    plt.figure(figsize=(8, 6))
    plt.scatter(code_vectors[:, 0], code_vectors[:, 1], c=lab, cmap=cmap, alpha=alpha)
    plt.colorbar()
    plt.grid()
    plt.savefig("{0}".format(plot_fname), dpi=300)
    plt.clf()
