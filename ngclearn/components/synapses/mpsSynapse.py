from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.matrix_utils import decompose_to_mps
from ngcsimlib.logger import info

from ngclearn import compilable
from ngclearn import Compartment

class MPSSynapse(JaxComponent):
    """
    A Matrix Product State (MPS) compressed synaptic cable.

    This component represents a synaptic weight matrix decomposed into a 
    contracted chain of low-rank tensor cores (also known as a Tensor Train). 
    This architecture drastically reduces parameter counts for high-dimensional 
    layers—from O(N*M) to O(N*K + M*K)—while maintaining high expressive power
    and biological plausibility through local error-driven updates.

    | References:
    | Stoudenmire, E. Miles, and David J. Schwab. "Supervised learning with 
    | quantum-inspired tensor networks." Advances in neural information 
    | processing systems 29 (2016).
    |
    | Novikov, Alexander, et al. "Tensorizing neural networks." Advances in 
    | neural information processing systems 28 (2015).
    |
    | Nuijten, W. W. L., et al. "A Message Passing Realization of Expected 
    | Free Energy Minimization." arXiv preprint arXiv:2501.03154 (2025).
    |
    | Wilson, P. "Performing Active Inference with Explainable Tensor 
    | Networks." (2024).
    |
    | Fields, Chris, et al. "Control flow in active inference systems." 
    | arXiv preprint arXiv:2303.01514 (2023).

    | --- Synapse Compartments: ---
    | inputs - external input signal values (shape: batch_size x in_dim)
    | outputs - transformed signal values (shape: batch_size x out_dim)
    | pre - pre-synaptic latent state values for learning (shape: batch_size x in_dim)
    | post - post-synaptic error signal values for learning (shape: batch_size x out_dim)
    | core1 - first MPS tensor core (shape: 1 x in_dim x bond_dim)
    | core2 - second MPS tensor core (shape: bond_dim x out_dim x 1)
    | key - JAX PRNG key used for stochasticity

    Args:
        name: the string name of this component

        shape: tuple specifying the shape of the latent synaptic weight matrix
            (number of inputs, number of outputs)

        bond_dim: the internal rank or "bond dimension" of the MPS compression. 
            Higher values increase expressive power at the cost of more parameters.
            (Default: 16)

        batch_size: the number of samples in a concurrent batch (Default: 1)
    """

    def __init__(self, name, shape, bond_dim=16, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        self.shape = shape
        self.bond_dim = bond_dim

        # Initialize synaptic cores using a small normal distribution
        tmp_key, *subkeys = random.split(self.key.get(), 3)

        # Core 1: maps input dimension to the internal bond dimension
        c1 = random.normal(subkeys[0], (1, shape[0], bond_dim)) * 0.05
        self.core1 = Compartment(c1)

        # Core 2: maps internal bond dimension to the output dimension
        c2 = random.normal(subkeys[1], (bond_dim, shape[1], 1)) * 0.05
        self.core2 = Compartment(c2)

        # Initialize Port/Compartment values
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))

        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.pre = Compartment(preVals)
        self.post = Compartment(postVals)

    @compilable
    def advance_state(self):
        """
        Performs the forward synaptic transformation using MPS contraction.
        
        The full transformation is equivalent to: outputs = inputs @ (Core1 * Core2),
        but computed via iterative contraction to maintain memory efficiency:
        1. z = inputs contracted with Core1 (Batch x Bond_Dim)
        2. outputs = z contracted with Core2 (Batch x Out_Dim)
        """
        x = self.inputs.get()
        c1 = self.core1.get()
        c2 = self.core2.get()

        # Contraction 1: (Batch, In) @ (1, In, Bond) -> (Batch, Bond)
        z = jnp.einsum('bi,mik->bk', x, c1)

        # Contraction 2: (Batch, Bond) @ (Bond, Out, 1) -> (Batch, Out)
        out = jnp.einsum('bk,kno->bn', z, c2)

        self.outputs.set(out)

    @compilable
    def project_backward(self, error_signal):
        """
        Projects an error signal backwards through the compressed synaptic cable.
        
        This allows the passing of messages/gradients through the hierarchy 
        without ever reconstructing the full dense weight matrix, ensuring 
        O(N) complexity relative to the input/output dimensions.
        """
        c1 = self.core1.get()
        c2 = self.core2.get()
        # 1. Project error through Core 2 to the bond space
        e_back = jnp.einsum('bo,kno->bk', error_signal, c2)
        # 2. Project from bond space through Core 1 to the input space
        return jnp.einsum('bk,mik->bi', e_back, c1)

    @compilable
    def evolve(self, eta=0.01):
        """
        Updates the MPS tensor cores using local error-driven (Hebbian) gradients.
        
        This utilizes the 'pre' and 'post' compartments to update core1 and core2. 
        Because the weights are factorized, the update to each core depends on 
        the activity and errors projected through the other core, maintaining
        global consistency through local message passing.
        """
        x = self.pre.get()       # Shape: (Batch, In)
        err = self.post.get()    # Shape: (Batch, Out)
        c1 = self.core1.get()    # Shape: (1, In, K)
        c2 = self.core2.get()    # Shape: (K, Out, 1)

        # 1. Compute latent bond activity for Core 2 update
        z = jnp.einsum('bi,mik->bk', x, c1)

        # 2. Update Core 2 (Gradients relative to bond activity and output error)
        dc2 = jnp.einsum('bk,bn->kn', z, err)
        dc2 = jnp.expand_dims(dc2, axis=2) 

        # 3. Update Core 1 (Gradients relative to input activity and back-projected error)
        err_back = jnp.einsum('bn,kno->bk', err, c2) 
        dc1 = jnp.einsum('bi,bk->ik', x, err_back)
        dc1 = jnp.expand_dims(dc1, axis=0) 

        # Apply updates to synaptic cores
        self.core1.set(c1 + eta * dc1)
        self.core2.set(c2 + eta * dc2)

    @compilable
    def reset(self):
        """
        Resets input, output, and activity compartments to zero.
        """
        if not self.inputs.targeted:
            self.inputs.set(jnp.zeros((self.batch_size, self.shape[0])))

        self.outputs.set(jnp.zeros((self.batch_size, self.shape[1])))
        
        if not self.pre.targeted:
            self.pre.set(jnp.zeros((self.batch_size, self.shape[0])))

        if not self.post.targeted:
            self.post.set(jnp.zeros((self.batch_size, self.shape[1])))

    @property
    def weights(self):
        """
        Reconstructs the full dense matrix from the MPS cores for analysis.
        Note: This is computationally expensive for high-dimensional layers.
        """
        return Compartment(jnp.einsum('mik,kno->in', self.core1.get(), self.core2.get()))

    @weights.setter
    def weights(self, W):
        """
        Sets the synaptic cores by decomposing a provided dense matrix W
        using Singular Value Decomposition (SVD).
        """
        c1, c2 = decompose_to_mps(W, bond_dim=self.bond_dim)
        self.core1.set(c1)
        self.core2.set(c2)

    @classmethod
    def help(cls):
        """
        Returns an info dictionary describing the component.
        """
        properties = {
            "synapse_type": "MPSSynapse - performs a compressed synaptic "
                            "transformation of inputs to produce output signals via "
                            "Matrix Product State (MPS) core contractions."
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic latent state values for learning",
                 "post": "Post-synaptic error signal values for learning"},
            "states":
                {"core1": "First MPS tensor core (1, in_dim, bond_dim)",
                 "core2": "Second MPS tensor core (bond_dim, out_dim, 1)",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of compressed synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of latent weight matrix (in_dim, out_dim)",
            "bond_dim": "The compression rank/bond-dimension of the MPS chain",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [inputs @ Core1] @ Core2",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("MPS_Test") as ctx:
        Wab = MPSSynapse("Wab", (10, 5), bond_dim=4)
    print(Wab)