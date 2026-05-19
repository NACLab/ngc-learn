import jax
from ngcsimlib import deprecate_args
from ngclearn.utils.analysis.probe import Probe
from jax import jit, random, numpy as jnp, lax, nn
from functools import partial as bind
from ngclearn.utils.metric_utils import measure_ARI

@bind(jax.jit, static_argnums=[2])
def _run_kmeans_probe(_embeddings, centroids, n_clusters):
    ## Broadcast distances: (n_samples, 1, n_features) - (1, n_clusters, n_features)
    distances = jnp.sum((_embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    labels_pred = jnp.argmin(distances, axis=1)
    ## Re-estimate centroids/means
    one_hot_preds = labels_pred[:, None] == jnp.arange(n_clusters)
    counts = jnp.maximum(one_hot_preds.sum(axis=0, keepdims=True).T, 1.0)
    centroids = jnp.dot(one_hot_preds.T.astype(jnp.float32), _embeddings) / counts
    return centroids

@bind(jax.jit, static_argnums=[2])
def _predict_with_probe(_embeddings, centroids, n_clusters):
    ## Final pass to compute stable predictions
    distances = jnp.sum((_embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    labels_pred = jnp.argmin(distances, axis=1)
    Y_pred = nn.one_hot(labels_pred, n_clusters)
    return labels_pred, Y_pred

class KMeansProbe(Probe):
    """
    This implements a K-means clustering probe, which is useful for evaluating the quality of
    encodings/embeddings in light of the ability to cluster downstream data. Currently, this
    probe only supports L2/Euclidean distance-based clustering.

    Args:
        dkey: init seed key

        source_seq_length: length of input sequence (e.g., height x width of the image feature)

        input_dim: input dimensionality of probe

        out_dim: output dimensionality of probe - number of clusters for this probe to create

        batch_size: <Unused>

    """

    def __init__(
            self,
            dkey,
            source_seq_length,
            input_dim,
            out_dim=2, ## number of clusters/centroids to uncover
            batch_size=1,
            **kwargs
    ):
        super().__init__(dkey, batch_size, **kwargs)
        self.dkey, *subkeys = random.split(self.dkey, 3)
        self.source_seq_length = source_seq_length
        self.input_dim = input_dim
        self.n_clusters = self.out_dim = out_dim
        ## centroids that will be uncovered by this probe
        self.centroids : jax.Array = None

    def _init(self, embeddings):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))
        ## choose random data-points to serve as centroids at iteration 0
        self.dkey, *subkeys = random.split(self.dkey, 15)
        n_samples, n_features = _embeddings.shape
        random_indices = random.choice(
            subkeys[0], n_samples, shape=(self.n_clusters,), replace=False
        )
        self.centroids = _embeddings[random_indices]

    def process(self, embeddings, dkey=None):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))
        ## Compute final geometric vs semantic conformity via ARI
        _, Y_pred = _predict_with_probe(_embeddings, self.centroids, self.n_clusters)
        return Y_pred ## (B, C)

    def update(self, embeddings, labels, dkey=None):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))
        self.centroids = _run_kmeans_probe(_embeddings, self.centroids, self.n_clusters)
        L = 0. ## FIXME: should be clustering loss
        predictions = self.process(_embeddings)
        return L, predictions

    def fit(self, dataset, dev_dataset=None, n_iter=20, patience=20):
        data, labels = dataset
        _labels = jnp.argmax(labels, axis=-1)

        self._init(data) ## init K-means centroids
        ari = 0.
        for i in range(n_iter): ## Run vectorized K-Means optimization loop
            _L, py = self.update(data, labels)
            labels_pred = jnp.argmax(py, axis=1)
            ari_i = measure_ARI(_labels, labels_pred)
            print(f"\r{i}: ARI = {ari_i}", end="")
            if ari_i > ari:
                ari = ari_i
        print()
        return ari
