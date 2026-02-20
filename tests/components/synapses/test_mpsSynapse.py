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
    print(f"MPS Reconstruction Test Passed. Error: {error*100:.2f}%")

def test_mps_synapse_learning():
    """
    Tests if MPSSynapse can learn from error signals via the evolve method.
    """
    dkey = random.PRNGKey(123)
    in_dim, out_dim = 10, 5
    bond_dim = 4
    
    with Context("mps_learning_test") as ctx:
        mps = MPSSynapse("mps", (in_dim, out_dim), bond_dim=bond_dim)
        advance = MethodProcess("advance") >> mps.advance_state
        evolve = MethodProcess("evolve") >> mps.evolve
        reset = MethodProcess("reset") >> mps.reset

    # Target: Map x to target_y
    x = random.normal(dkey, (1, in_dim))
    target_y = random.normal(dkey, (1, out_dim))
    
    # 1. Initial prediction
    mps.inputs.set(x)
    advance.run()
    initial_y = mps.outputs.get()
    initial_error = jnp.sum((target_y - initial_y)**2)
    
    # 2. Learning step
    # Hebbian learning expects pre and post
    # error = target - predicted
    error_signal = (target_y - initial_y)
    
    mps.pre.set(x)
    mps.post.set(error_signal)
    
    # Run learning multiple times to see descent
    for _ in range(5):
        evolve.run(eta=0.1)
        # Re-predict
        advance.run()
        current_y = mps.outputs.get()
        current_error = jnp.sum((target_y - current_y)**2)
        mps.post.set(target_y - current_y) # Update error signal for next step
    
    final_y = mps.outputs.get()
    final_error = jnp.sum((target_y - final_y)**2)
    
    print(f"Initial Error: {initial_error:.6f}")
    print(f"Final Error:   {final_error:.6f}")
    
    assert final_error < initial_error
    print("MPS Learning Test Passed.")

if __name__ == "__main__":
    test_mps_synapse_reconstruction()
    test_mps_synapse_learning()
