from jax import numpy as jnp, random
import numpy as np
from ngcsimlib.context import Context
from ngclearn import MethodProcess
from ngclearn.components.synapses.mpsSynapse import MPSSynapse

def test_mps_synapse_reconstruction():
    """
    Tests if MPSSynapse can be initialized from a dense matrix and 
    reproduce the transformation reasonably well.
    """
    dkey = random.PRNGKey(42)
    in_dim, out_dim = 20, 10
    bond_dim = 8
    
    with Context("mps_test") as ctx:
        mps = MPSSynapse("mps", (in_dim, out_dim), bond_dim=bond_dim)
        advance = MethodProcess("advance") >> mps.advance_state
        reset = MethodProcess("reset") >> mps.reset

    # Create a structured matrix
    W_orig = random.normal(dkey, (in_dim, out_dim))
    
    # Set weights via setter (uses decompose_to_mps)
    mps.weights = W_orig
    
    # Check reconstruction fidelity
    W_recon = mps.weights
    error = jnp.linalg.norm(W_orig - W_recon) / jnp.linalg.norm(W_orig)
    
    # With bond_dim=8 and rank=10, error should be small but non-zero
    assert error < 0.5 
    
    # Test transformation correctness
    x = random.normal(dkey, (1, in_dim))
    mps.inputs.set(x)
    advance.run()
    
    y_mps = mps.outputs.get()
    y_dense = x @ W_recon
    
    np.testing.assert_allclose(y_mps, y_dense, atol=1e-5)
    print(f"MPS Test Passed. Reconstruction Error: {error*100:.2f}%")

if __name__ == "__main__":
    test_mps_synapse_reconstruction()
