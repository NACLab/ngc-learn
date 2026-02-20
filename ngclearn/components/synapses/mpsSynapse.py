import jax.numpy as jnp
from jax import random, lax
from ngclearn import compilable
from ngclearn.components import JaxComponent
from ngclearn.utils import Compartment

def decompose_to_mps(W, bond_dim=16):
    """
    Utility to decompose a dense matrix W into two MPS cores using SVD.
    
    Args:
        W: dense matrix of shape (in_dim, out_dim)
        bond_dim: the internal rank of the MPS compression
        
    Returns:
        core1 of shape (1, in_dim, bond_dim) and core2 of shape (bond_dim, out_dim, 1)
    """
    U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
    k = min(bond_dim, len(S))
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    
    s_sqrt = jnp.sqrt(S_k)
    core1 = (U_k * s_sqrt).reshape(1, W.shape[0], k)
    core2 = (s_sqrt[:, None] * Vh_k).reshape(k, W.shape[1], 1)
    return core1, core2

class MPSSynapse(JaxComponent):
    """
    A Matrix Product State (MPS) compressed synaptic cable.
    Instead of maintaining a dense weight matrix, this component represents
    the synaptic transformation through a contracted chain of low-rank tensor cores,
    drastically reducing parameter count for high-dimensional layers.
    
    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | core1 - first MPS tensor core (1 x in_dim x bond_dim)
    | core2 - second MPS tensor core (bond_dim x out_dim x 1)
    
    Args:
        name: the string name of this component
        shape: tuple specifying shape (in_dim, out_dim)
        bond_dim: the internal rank of the MPS compression (Default: 16)
        batch_size: batch size dimension (Default: 1)
    """
    def __init__(self, name, shape, bond_dim=16, batch_size=1, **kwargs):
        super().__init__(name, batch_size=batch_size, **kwargs)
        
        self.shape = shape
        self.bond_dim = bond_dim
        
        # Initialize cores randomly or from a specific distribution
        key, *subkeys = random.split(self.key.get(), 3)
        
        # Core 1: (1, In, K)
        c1 = random.normal(subkeys[0], (1, shape[0], bond_dim)) * 0.05
        self.core1 = Compartment(c1)
        
        # Core 2: (K, Out, 1)
        c2 = random.normal(subkeys[1], (bond_dim, shape[1], 1)) * 0.05
        self.core2 = Compartment(c2)
        
        # Interaction Ports
        self.inputs = Compartment(jnp.zeros((batch_size, shape[0])))
        self.outputs = Compartment(jnp.zeros((batch_size, shape[1])))

    @compilable
    def advance_state(self, t, dt):
        """
        Performs the MPS contraction transformation: outputs = (inputs @ Core1) @ Core2
        """
        x = self.inputs.get()
        c1 = self.core1.get()
        c2 = self.core2.get()
        
        # Step 1: Contract input with the first core
        # (Batch, In) @ (1, In, K) -> (Batch, K)
        z = jnp.einsum('bi,mik->bk', x, c1)
        
        # Step 2: Contract result with the second core
        # (Batch, K) @ (K, Out, 1) -> (Batch, Out)
        out = jnp.einsum('bk,kno->bo', z, c2)
        
        self.outputs.set(out)

    @compilable
    def reset(self):
        """
        Resets input and output compartments to zero.
        """
        self.inputs.set(jnp.zeros((self.batch_size, self.shape[0])))
        self.outputs.set(jnp.zeros((self.batch_size, self.shape[1])))

    @property
    def weights(self):
        """
        Reconstructs the full dense matrix from MPS cores for analysis.
        """
        return jnp.einsum('mik,kno->in', self.core1.get(), self.core2.get())

    @weights.setter
    def weights(self, W):
        """
        Sets the synaptic cores by decomposing a provided dense matrix W.
        """
        c1, c2 = decompose_to_mps(W, bond_dim=self.bond_dim)
        self.core1.set(c1)
        self.core2.set(c2)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "MPSSynapse - performs a compressed synaptic "
                            "transformation of inputs to produce output signals via "
                            "Matrix Product State (MPS) core contractions."
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"core1": "First MPS tensor core (1, in_dim, bond_dim)",
                 "core2": "Second MPS tensor core (bond_dim, out_dim, 1)"},
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
