import jax.numpy as jnp

def decompose_to_mps(W, bond_dim=16):
    """
    Decomposes a dense matrix W into two MPS cores using SVD.

    Args:
        W: The dense matrix to decompose of shape (in_dim, out_dim).

        bond_dim: The internal rank/bond-dimension of the MPS compression.

    Returns:
        A tuple containing:
            core1: First tensor core of shape (1, in_dim, bond_dim).
            core2: Second tensor core of shape (bond_dim, out_dim, 1).
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
