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
        """
        Runs this probe's inference scheme over a pool of data.

        Args:
            data: a dataset or design tensor/matrix containing encoding vector sequences; shape (N, T, embed_dim) or (N, embed_dim)

        Returns: 
            the output scores/predictions made by this probe
        """
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

    def fit(self, dataset, dev_dataset=None, n_iter=50, patience=20):
        """
        Fits this probe to a pool of data.

        Args:
            dataset: a dataset tuple containing two design tensors/matrices (X, Y), with the first containing encoding
                vector sequences of shape (N, T, embed_dim) or (N, embed_dim) and the second containing the
                corresponding labels/targets for the embedding data of shape (N, target_dim); (Default: None)

            dev_dataset: an optional development set tuple, with same format as `dataset` (Default: None)

            n_iter: number of iterations to run model fitting (Default: 50 iterations)

            patience: number of iterations of improvement (decrease) in loss before early-stopping enacted
            
        Returns:
            the output scores/predictions made by this probe
        """
        data, labels = dataset
        dev_data = dev_labels = None
        if dev_dataset is not None:
            dev_data, dev_labels = dev_dataset

        _data = data
        if len(_data.shape) < 3:
            _data = jnp.expand_dims(_data, axis=1)

        n_samples, seq_len, dim = _data.shape
        size_modulo = n_samples % self.batch_size
        if size_modulo > 0: 
            ## we append some dup data for dataset design tensors that do not divide by batch size evenly
            _chunk = _data[0:size_modulo, :, :]
            _data = jnp.concatenate((_data, _chunk), axis=0)
            n_samples, seq_len, dim = _data.shape
        n_batches = int(n_samples / self.batch_size)

        ## run main probe fitting loop
        impatience = 0
        final_L = 10000.
        best_acc = 0.
        #Y_mu = []
        _Y = None
        for ii in range(n_iter):
            ## shuffle data (to ensure i.i.d. across sequences)
            self.dkey, *subkeys = random.split(self.dkey, 2)
            ptrs = random.permutation(subkeys[0], n_samples)
            _X = _data[ptrs, :, :]
            _Y = labels[ptrs, :]
            ## run one epoch over data tensors
            L = 0.
            acc = 0.
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
                acc = jnp.sum(jnp.equal(jnp.argmax(py, axis=1), jnp.argmax(y_mb, axis=1))) + acc
                L = (_L * x_mb.shape[0]) + L ## we remove the batch division from loss w.r.t. x_mb/y_mb
                if dev_data is not None:
                    print(f"\r{ii} L = {L / Ns:.3f} Acc = {acc / Ns:.2f}  Dev.Acc = {best_acc:.2f}", end="")
                else:
                    print(f"\r{ii} L = {L / Ns:.3f} Acc = {acc / Ns:.2f}", end="")
                # if ii == ii-1:
                #     Y_mu.append(py)
            print()
            acc = acc / Ns
            final_L = L / Ns ## compute current loss over (train) dataset
            # if ii == ii - 1:
            #     Y_mu = jnp.concatenate(Y_mu, axis=0)

            impatience += 1
            if dev_data is not None:
                Ymu = self.predict(dev_data)
                acc = jnp.sum(jnp.equal(jnp.argmax(Ymu, axis=1), jnp.argmax(dev_labels, axis=1))) / (dev_labels.shape[0] * 1.)
                if acc > best_acc:
                    best_acc = acc
                    impatience = 0
            else: ## use training acc if no dev-set provided
                if acc > best_acc:
                    best_acc = acc
                    impatience = 0

            if impatience > patience:
                break  ## execute early stopping
        return final_L

