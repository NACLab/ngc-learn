from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np

np.random.seed(42)
from ngclearn.components import HodgkinHuxleyCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_almost_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch

import matplotlib.pyplot as plt


def test_hodgkinHuxleyCell1():
    name = "hh_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 0.01  # 1.  # ms

    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = HodgkinHuxleyCell(
            name="a", n_units=1, tau_v=1., resist_m=1., key=subkeys[0]
        )

        # """
        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(advance_process.pure, name="run")
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (Process("reset_proc")
                         >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
        # """

        """
        reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
        ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
        advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
        ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")
        """

        ## set up non-compiled utility commands
        @Context.dynamicCommand
        def clamp(x):
            a.j.set(x)

    ## input spike train
    x_seq = jnp.zeros((1, 20))
    y_seq = jnp.array(
        [[
            0.02414415, 0.04820144, 0.07217567, 0.09607048, 0.11988933, 0.14363553, 0.16731221, 0.19092241,
            0.21446899,  0.23795472, 0.26138224, 0.28475408, 0.30807265, 0.3313403,  0.35455925, 0.37773165,
            0.40085957, 0.42394499, 0.44698984, 0.46999594]], dtype=jnp.float32)

    v = []
    ctx.reset()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
        ctx.clamp(x_t)
        ctx.run(t=ts * 1., dt=dt)
        v.append(a.v.value[0, 0])
    outs = jnp.array(v)
    diff = np.abs(outs - y_seq)
    ## delta/error should be approximately zero
    assert_array_almost_equal(diff, diff * 0., decimal=6)

#test_hodgkinHuxleyCell1()
