from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import VarTrace
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch

def test_varTrace1():
    name = "trace_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    trace_increment = 0.1
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = VarTrace(
            name="a", n_units=1, a_delta=trace_increment, decay_type="step", tau_tr=1., 
            key=subkeys[0]
        )

        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (Process("reset_proc")
                         >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

        ## set up non-compiled utility commands
        @Context.dynamicCommand
        def clamp(x):
            a.inputs.set(x)

    ## input spike train
    x_seq = jnp.asarray([[1., 1., 0., 0., 1.]], dtype=jnp.float32)
    ## desired output pulses
    y_seq = x_seq * trace_increment 

    outs = []
    ctx.reset()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
        ctx.clamp(x_t)
        ctx.run(t=ts * 1., dt=dt)
        outs.append(a.outputs.value)
    outs = jnp.concatenate(outs, axis=1)
    #print(outs)

    ## output should equal input
    assert_array_equal(outs, y_seq)

#test_varTrace1()
