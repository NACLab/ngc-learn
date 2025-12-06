from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.convolution.traceSTDPDeconvSynapse import TraceSTDPDeconvSynapse
from numpy.testing import assert_array_equal

def test_TraceSTDPDeconvSynapse1():
    name = "stdp_deconv_ctx"
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
        a = TraceSTDPDeconvSynapse(
            "a", shape, (x_size, x_size), A_plus=1., A_minus=1., eta=0., filter_init=dist.constant(value=1.),
            # bias_init=None,
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

    ## fake out a mix of pre-synaptic spikes/no-spikes
    x = np.ones(x_shape)
    x[0, 0, 0, 0] = 0.
    x[0, 1, 1, 0] = 0.

    y_truth = jnp.array(
        [[[[0.], [1.]],
          [[1.], [1.]]]]
    )

    reset_process.run() #ctx.reset()
    a.inputs.set(x)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1., dt=dt)
    y = (a.outputs.get() > 0.) * 1. ## fake out post-syn spikes
    assert_array_equal(y, y_truth)
    # print(y)
    # print("y.Tr:\n", y_truth)
    # print("======")

    # print("NGC-Learn.shape = ", node.outputs.get().shape)
    a.preSpike.set(x)
    a.postSpike.set(y)
    a.preTrace.set(x * 0.4)  ## fake out pre-syn trace values
    a.postTrace.set(y * 1.3)  ## fake out post-syn trace values
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1., dt=dt)
    dK = a.dWeights.get()
    backtransmit_process.run(t=1., dt=dt)  # ctx.backtransmit(t=1., dt=dt)
    dx = a.dInputs.get()
    dK_truth = jnp.array(
        [[[[0.]],
          [[-0.9]]],
         [[[-0.9]],
          [[-1.8]]]]
    )
    dx_truth = jnp.array(
        [[[[3.],
           [2.]],
          [[2.],
            [1.]]]]
    )
    # print(dK)
    # print("dK.Tr:\n", dK_truth)
    # print(dx)
    # print("dx.Tr:\n", dx_truth)
    assert_array_equal(dK, dK_truth)
    assert_array_equal(dx, dx_truth)

#test_TraceSTDPDeconvSynapse1()
