from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import SLIFCell
from numpy.testing import assert_array_equal

from ngclearn import MethodProcess, Context


def test_sLIFCell1():
    name = "slif_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    trace_increment = 0.1
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = SLIFCell(
            name="a", n_units=1, tau_m=50., resist_m=10., thr=0.3, key=subkeys[0]
        )

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)
        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

        ## set up non-compiled utility commands
        def clamp(x):
            a.j.set(x)

    ## input spike train
    x_seq = jnp.asarray([[1., 1., 0., 0., 1., 1., 0.]], dtype=jnp.float32)
    ## desired output/epsp pulses
    y_seq = jnp.asarray([[0., 1., 0., 0., 0., 1., 0.]], dtype=jnp.float32)

    outs = []
    reset_process.run()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
        clamp(x_t)
        advance_process.run(t=ts * 1., dt=dt)
        outs.append(a.s.get())
    outs = jnp.concatenate(outs, axis=1)

    ## output should equal input
    assert_array_equal(outs, y_seq)

#test_sLIFCell1()
