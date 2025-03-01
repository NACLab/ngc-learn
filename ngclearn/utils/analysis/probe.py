from jax import random, numpy as jnp

class Probe():
    """
    General framework for an analysis probe (that may or may not be learnable in an iterative fashion).

    Args:
        dkey: init seed key

        batch_size: size of batches to process per internal call to update (or process)

    """
    def __init__(
            self, dkey, batch_size=4, **kwargs
    ):
        #dkey, *subkeys = random.split(dkey, 3)
        self.dkey = dkey
        self.batch_size = batch_size

    def process(self, embeddings):
        predictions = None
        return predictions

    def update(self, embeddings, labels):
        L = predictions = None
        return L, predictions

    def predict(self, data):
        _data = data
        if len(_data.shape) < 3:
            _data = jnp.expand_dims(_data, axis=1)

        n_samples, seq_len, dim = _data.shape
        n_batches = int(n_samples / self.batch_size)
        s_ptr = 0
        e_ptr = self.batch_size
        Y_mu = []
        for b in range(n_batches):
            x_mb = _data[s_ptr:e_ptr, :, :]  ## slice out 3D batch tensor
            s_ptr = e_ptr
            e_ptr += x_mb.shape[0]
            y_mu = self.process(x_mb)
            Y_mu.append(y_mu)
        Y_mu = jnp.concatenate(Y_mu, axis=0)
        return Y_mu

    def fit(self, data, labels, n_iter=50):
        _data = data
        if len(_data.shape) < 3:
            _data = jnp.expand_dims(_data, axis=1)

        n_samples, seq_len, dim = _data.shape
        n_batches = int(n_samples / self.batch_size)

        Y_mu = []
        _Y = None
        for iter in range(n_iter):
            ## shuffle data (to ensure i.i.d. across sequences)
            self.dkey, *subkeys = random.split(self.dkey, 2)
            ptrs = random.permutation(subkeys[0], n_samples)
            _X = _data[ptrs, :, :]
            _Y = labels[ptrs, :]
            ## run one epoch over data tensors
            L = 0.
            Ns = 0.

            s_ptr = 0
            e_ptr = self.batch_size
            for b in range(n_batches):
                x_mb = _X[s_ptr:e_ptr, :, :] ## slice out 3D batch tensor
                y_mb = _Y[s_ptr:e_ptr, :]
                s_ptr = e_ptr
                e_ptr += x_mb.shape[0]
                Ns += x_mb.shape[0]

                _L, py = self.update(x_mb, y_mb)
                L = _L + L
                print(f"\r{iter} L = {L/Ns}", end="") #  p(y|z):\n{py}")
                if iter == n_iter-1:
                    Y_mu.append(py)
            print()
            if iter == n_iter - 1:
                Y_mu = jnp.concatenate(Y_mu, axis=0)
        return Y_mu, _Y
