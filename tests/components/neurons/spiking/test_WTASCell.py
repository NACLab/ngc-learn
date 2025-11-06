from jax import numpy as jnp, random, jit
import numpy as np

np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.components.neurons.spiking.WTASCell import WTASCell
from numpy.testing import assert_array_equal


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
        advance_process = (MethodProcess(name="advance_proc")
                           >> a.advance_state)
        #ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (MethodProcess(name="reset_proc")
                         >> a.reset)
        #ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
        #"""

        # ## set up non-compiled utility commands
        # @Context.dynamicCommand
        # def clamp(x):
        #     a.j.set(x)

    def clamp(x):
        a.j.set(x)

    ## input spike train
    x_seq = jnp.asarray([[0., 1.], [0., 1.], [1., 0.], [1., 0.]], dtype=jnp.float32)
    ## desired output/epsp pulses
    y_seq = x_seq

    outs = []
    reset_process.run()
    for ts in range(x_seq.shape[0]):
        x_t = x_seq[ts:ts+1, :]  ## get data at time t
        clamp(x_t) #ctx.clamp(x_t)
        advance_process.run(t=ts * 1., dt=dt)
        outs.append(a.s.get())
    outs = jnp.concatenate(outs, axis=0)
    # print(outs)
    # print(y_seq)
    #exit()
    ## output should equal input
    assert_array_equal(outs, y_seq)

#test_WTASCell1()
