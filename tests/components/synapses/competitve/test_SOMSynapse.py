from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.competitive.SOMSynapse import SOMSynapse
from numpy.testing import assert_array_equal

def test_SOMSynapse_batch_size_1():
    seed = 1234
    in_dim = 15
    lr = 0.05
    height = 15
    width = 15
    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 3)
    with Context("som_ctx") as som_ctx:
        som = SOMSynapse(
            name="a",
            n_inputs=in_dim,
            n_units_x=height,
            n_units_y=width,
            eta=lr, ## initial learning-rate (alpha)
            distance_function="euclidean",
            neighbor_function="ricker",
            weight_init=dist.uniform(0., 1.),
            key=subkeys[0]
        )
        evolve_process = (MethodProcess("evolve_process")
                          >> som.evolve)
        advance_process = (MethodProcess("advance_proc")
                          >> som.advance_state)
        reset_process = (MethodProcess("reset_proc")
                        >> som.reset)

    W = som.weights.get()
    print(f"SOM.radius(0): {som.radius.get()[0,0]:.5f}  "
      f"SOM.eta(0): {som.eta.get()[0,0]:.5f}  "
      f"W.n(0): {jnp.linalg.norm(W)}")


    x_n = np.random.normal(size=(1, in_dim)) # random input (B, in_dim)
    reset_process.run()
    som.inputs.set(x_n) ## clamp input to SOM input layer
    advance_process.run(t=1., dt=1.)
    evolve_process.run(t=1., dt=1.)

    i_tick = som.i_tick.get() ## each sample/update counts as an "up-tick"
    print(f"\r{som.i_tick.get()} samps; "
          f"SOM.radius: {som.radius.get()[0,0]:.5f}  "
          f"SOM.eta: {som.eta.get()[0,0]:.5f}", end="")


test_SOMSynapse_batch_size_1()


def test_SOMSynapse_batch_size_n():
    B = 4 # batch size
    seed = 1234
    in_dim = 15
    lr = 0.05
    height = 15
    width = 15
    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 3)
    ## set up a single layer SOM
    with Context("som_ctx") as som_ctx:
        som = SOMSynapse(
            name="a",
            n_inputs=in_dim,
            n_units_x=height,
            n_units_y=width,
            eta=lr, ## initial learning-rate (alpha)
            distance_function="euclidean",
            neighbor_function="ricker",
            weight_init=dist.uniform(0., 1.),
            batch_size=B,
            key=subkeys[0]
        )
        evolve_process = (MethodProcess("evolve_process")
                          >> som.evolve)
        advance_process = (MethodProcess("advance_proc")
                          >> som.advance_state)
        reset_process = (MethodProcess("reset_proc")
                        >> som.reset)

    W = som.weights.get()
    print(f"SOM.radius(0): {som.radius.get()[0,0]:.5f}  "
      f"SOM.eta(0): {som.eta.get()[0,0]:.5f}  "
      f"W.n(0): {jnp.linalg.norm(W)}")


    x_n = np.random.normal(size=(B, in_dim)) # random input (B, in_dim)
    reset_process.run()
    som.inputs.set(x_n) ## clamp input to SOM input layer
    advance_process.run(t=1., dt=1.)
    evolve_process.run(t=1., dt=1.)

    assert som.bmu.get().shape == (B, 1)
    assert som.neighbor_weights.get().shape == (B, height * width)
    assert som.delta.get().shape == (B, in_dim, height * width)
    assert som.outputs.get().shape == (B, height * width)
    assert som.dWeights.get().shape == (in_dim, height * width)

    i_tick = som.i_tick.get() ## each sample/update counts as an "up-tick"
    print(f"\r{som.i_tick.get()} samps; "
          f"SOM.radius: {som.radius.get()[0,0]:.5f}  "
          f"SOM.eta: {som.eta.get()[0,0]:.5f}", end="")

# test_SOMSynapse_batch_size_n()

