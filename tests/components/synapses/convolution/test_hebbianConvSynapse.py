from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.convolution.hebbianConvSynapse import HebbianConvSynapse
from numpy.testing import assert_array_equal

def test_HebbianConvSynapse1():
    name = "hebb_conv_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    padding_style = "SAME"
    stride = 1 #2
    batch_size = 1
    w_size = 2
    n_in_chan = 1 #4  # 1
    n_out_chan = 1 #5  # 1
    x_size = 2 #4

    shape = (w_size, w_size, n_in_chan, n_out_chan)
    x_shape = (batch_size, x_size, x_size, n_in_chan)

    # ---- build a simple Hebb-Conv system ----
    with Context(name) as ctx:
        a = HebbianConvSynapse(
            "a", shape, (x_size, x_size), eta=0., filter_init=dist.constant(value=1.), bias_init=None,
            stride=stride, padding=padding_style, batch_size=batch_size, key=subkeys[0]
        )

        evolve_process = (MethodProcess("evolve_process")
                          >> a.evolve)

        backtransmit_process = (MethodProcess("backtransmit_process")
                           >> a.backtransmit)

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    x = jnp.ones(x_shape)

    reset_process.run() # ctx.reset()
    a.inputs.set(x)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1., dt=dt)
    y = a.outputs.get()

    y_truth = jnp.array(
        [[[[4.],[2.]],
          [[2.], [1.]]]]
    )

    assert_array_equal(y, y_truth)
    # print(y)
    # print("y.Tr:\n", y_truth)
    # print("======")

    # print("NGC-Learn.shape = ", node.outputs.get().shape)
    a.pre.set(x)
    a.post.set(y)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1., dt=dt)
    dK = a.dWeights.get()
    backtransmit_process.run(t=1., dt=dt)  # ctx.backtransmit(t=1., dt=dt)
    dx = a.dInputs.get()
    dK_truth = jnp.array(
        [[[[9.]],
          [[6.]]],
         [[[6.]],
          [[4.]]]]
    )
    dx_truth = jnp.array(
        [[[[4.],
           [6.]],
          [[6.],
            [9.]]]]
    )
    # print(dK)
    # print("dK.Tr:\n", dK_truth)
    # print(dx)
    # print("dx.Tr:\n", dx_truth)
    assert_array_equal(dK, dK_truth)
    assert_array_equal(dx, dx_truth)

#test_HebbianConvSynapse1()
