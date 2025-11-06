from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.components.neurons.spiking.fitzhughNagumoCell import FitzhughNagumoCell
from numpy.testing import assert_array_equal


def test_fitzhughNagumoCell1():
    name = "fh_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 0.1 #1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = FitzhughNagumoCell(
            name="a", n_units=1, tau_m=1., resist_m=5., v_thr=2.1, key=subkeys[0]
        )

        # """
        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)
        # ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
        # """
        ## set up non-compiled utility commands
        # @Context.dynamicCommand
        # def clamp(x):
        #     a.j.set(x)

    def clamp(x):
        a.j.set(x)

    ## input spike train
    x_seq = jnp.asarray([[0., 0., 1., 1., 1., 1., 0., 0., 0., 0.]], dtype=jnp.float32)
    ## desired output/epsp pulses
    y_seq = jnp.asarray([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]], dtype=jnp.float32)

    outs = []
    reset_process.run()  # ctx.reset()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
        clamp(x_t)  # ctx.clamp(x_t)
        advance_process.run(t=ts * 1., dt=dt)  # ctx.run(t=ts * 1., dt=dt)
        outs.append(a.s.get())

    outs = jnp.concatenate(outs, axis=1)
    # print(outs)
    # print(y_seq)

    ## output should equal input
    assert_array_equal(outs, y_seq)

#test_fitzhughNagumoCell1()
