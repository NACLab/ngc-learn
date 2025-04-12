from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import TraceSTDPDeconvSynapse
import ngclearn.utils.weight_distribution as dist
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context

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

        #"""
        evolve_process = (Process("evolve_proc")
                           >> a.evolve)
        #ctx.wrap_and_add_command(evolve_process.pure, name="run")
        ctx.wrap_and_add_command(jit(evolve_process.pure), name="adapt")

        backtransmit_process = (Process("btransmit_proc")
                                >> a.backtransmit)
        ctx.wrap_and_add_command(jit(backtransmit_process.pure), name="backtransmit")

        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(advance_process.pure, name="run")
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (Process("reset_proc")
                         >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
        #"""

        """
        reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
        ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
        advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
        ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")
        evolve_cmd, evolve_args = ctx.compile_by_key(a, compile_key="evolve")
        ctx.add_command(wrap_command(jit(ctx.evolve)), name="adapt")
        backpass_cmd, backpass_args = ctx.compile_by_key(a, compile_key="backtransmit")
        ctx.add_command(wrap_command(jit(ctx.backtransmit)), name="backtransmit")
        """

    ## fake out a mix of pre-synaptic spikes/no-spikes
    x = np.ones(x_shape)
    x[0, 0, 0, 0] = 0.
    x[0, 1, 1, 0] = 0.

    y_truth = jnp.array(
        [[[[0.], [1.]],
          [[1.], [1.]]]]
    )

    ctx.reset()
    a.inputs.set(x)
    ctx.run(t=1., dt=dt)
    y = (a.outputs.value > 0.) * 1. ## fake out post-syn spikes
    assert_array_equal(y, y_truth)
    #print(y)
    #print("======")

    # print("NGC-Learn.shape = ", node.outputs.value.shape)
    a.preSpike.set(x)
    a.postSpike.set(y)
    a.preTrace.set(x * 0.4)  ## fake out pre-syn trace values
    a.postTrace.set(y * 1.3)  ## fake out post-syn trace values
    ctx.adapt(t=1., dt=dt)
    dK = a.dWeights.value
    #print(dK)
    ctx.backtransmit(t=1., dt=dt)
    dx = a.dInputs.value
    #print(dx)
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
    assert_array_equal(dK, dK_truth)
    assert_array_equal(dx, dx_truth)

#test_TraceSTDPDeconvSynapse1()
