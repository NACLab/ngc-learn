from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np

np.random.seed(42)
from ngclearn.components import WTASCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


def test_WTASCell1():
    name = "wtas_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = WTASCell(
            name="a", n_units=2, tau_m=25., resist_m=1., key=subkeys[0]
        )

        #"""
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
        """

        ## set up non-compiled utility commands
        @Context.dynamicCommand
        def clamp(x):
            a.j.set(x)

    ## input spike train
    x_seq = jnp.asarray([[0., 1.], [0., 1.], [1., 0.], [1., 0.]], dtype=jnp.float32)
    ## desired output/epsp pulses
    y_seq = x_seq

    outs = []
    ctx.reset()
    for ts in range(x_seq.shape[0]):
        x_t = x_seq[ts:ts+1, :]  ## get data at time t
        ctx.clamp(x_t)
        ctx.run(t=ts * 1., dt=dt)
        outs.append(a.s.value)
    outs = jnp.concatenate(outs, axis=0)
    #print(outs)
    #exit()
    ## output should equal input
    assert_array_equal(outs, y_seq)

#test_WTASCell1()
