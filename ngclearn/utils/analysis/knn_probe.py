import jax
import numpy as np
from ngclearn.utils.analysis.probe import Probe
from ngclearn.utils.model_utils import kwta
from jax import jit, random, numpy as jnp, lax, nn
from functools import partial as bind
from ngclearn.utils.distribution_generator import DistributionGenerator

@bind(jax.jit, static_argnums=[2, 3])
def _run_knn_probe(_embeddings, Wx, K, dist_fx=1):
    ## Notes:
    ### We do some 3D tensor math to handle a batch of predictions that need to be made
    ### B = batch-size, D = embedding/input dim, C = number classes, N = number of memories
    _Wx = jnp.expand_dims(Wx, axis=0)  ## 3D tensor format of KNN params (1 x N x D)
    embed_tensor = jnp.expand_dims(_embeddings, axis=1)  ## 3D projection of input signals (B x 1 x D)
    D = embed_tensor - _Wx  ## compute 3D batched delta tensor (B x N x D)
    ## get batched (negative) distance measurements
    dist = jnp.linalg.norm(D, ord=2, axis=2, keepdims=True)  ## (B x N x 1)
    if dist_fx == 2:
        dist = jnp.linalg.norm(D, ord=1, axis=2, keepdims=True)  ## (B x N x 1)
    ## else, default -> euclidean
    ### Note: negative distance allows us to find minimal points w/ maximal functions
    dist = -jnp.squeeze(dist, axis=2)  ## (B x N)
    ## now get K winners per sample in batch
    values, indices = lax.top_k(dist, K)
    return values, indices

class KNNProbe(Probe):
    """
    This implements a K-nearest neighbors (KNN) probe, which is useful for evaluating the quality of
    encodings/embeddings in light of some superivsory downstream data (e.g., label one-hot
    encodings or real-valued vector regression targets).

    Args:
        dkey: init seed key

        source_seq_length: length of input sequence (e.g., height x width of the image feature)

        input_dim: input dimensionality of probe

        out_dim: output dimensionality of probe

        batch_size: size of batches to process per internal call to update (or process)

        K: number of nearest neighbors to estimate output target

        dist_function: what distance function should be used in calculating nearest neighbors (Default: euclidean)

    """
    def __init__(
            self,
            dkey,
            source_seq_length,
            input_dim,
            out_dim,
            batch_size=1,
            K=1, ## number of nearest neighbors (K) to find
            dist_function="euclidean",
            **kwargs
    ):
        super().__init__(dkey, batch_size, **kwargs)
        self.dkey, *subkeys = random.split(self.dkey, 3)
        self.source_seq_length = source_seq_length
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.K = K
        self.dist_function = dist_function
        self.dist_fx = 0
        if self.dist_function == "manhattan":
            self.dist_fx = 1

        #flat_input_dim = input_dim * source_seq_length
        #W = jnp.zeros((flat_input_dim, out_dim))
        Wx = Wy = jnp.ones((1, 1)) ## Wy will be assumed to be one-hot encoded
        self.probe_params = (Wx, Wy)

    def process(self, embeddings, dkey=None): ## TODO: JIT-i-fy this
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))

        Wx, Wy = self.probe_params ## pull out KNN parameters
        values, indices =  _run_knn_probe(_embeddings, Wx, self.K, self.dist_fx)

        ## do K-neighbor voting scheme (find mode prediction)
        Y_counts = jnp.zeros((_embeddings.shape[0], Wy.shape[1]))
        for k in range(self.K):
            winner_k_indx = indices[:, k] ## batch of k-th winner of K winners
            Y_k = Wy[winner_k_indx, :] ## predicted Y's of k-th winner batch
            Y_counts = Y_counts + Y_k
        Y_pred = nn.one_hot(jnp.argmax(Y_counts, axis=1), num_classes=Wy.shape[1]) #, keepdims=True)
        return Y_pred ## (B, C)

    def update(self, embeddings, labels, dkey=None):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))

        ## a K-NN's learning phase is just storing the data internally directly
        Wx = _embeddings
        Wy = labels
        self.probe_params = (Wx, Wy)

if __name__ == '__main__':
    seed = 42
    D = 7
    C = 5
    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 3)
    knn = KNNProbe(
        subkeys[0], 1, input_dim=D, out_dim=C, K=1, dist_function="euclidean"
    )
    X = random.uniform(subkeys[1], shape=(10, D))
    Y = jnp.concat(
        [
            jnp.ones((2, C)) * jnp.array([[1., 0., 0., 0., 0.]]),
            jnp.ones((2, C)) * jnp.array([[0., 1., 0., 0., 0.]]),
            jnp.ones((2, C)) * jnp.array([[0., 0., 1., 0., 0.]]),
            jnp.ones((2, C)) * jnp.array([[0., 0., 0., 1., 0.]]),
            jnp.ones((2, C)) * jnp.array([[0., 0., 0., 0., 1.]])
         ],
        axis=0
    )
    knn.update(X, Y) ## fit KNN to data
    print(knn.process(X)) ## should construct the (smeared) identity matrix, exactly same as Y