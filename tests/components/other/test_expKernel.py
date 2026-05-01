from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import ExpKernel
from ngclearn import MethodProcess, Context

def test_expKernel1():
    name = "expKernel_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    trace_increment = 0.1
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = ExpKernel(
            name="a", n_units=1, dt=1., tau_w=500., nu=4., key=subkeys[0]
        )

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)
        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

        ## set up non-compiled utility commands
        def clamp(x):
            a.inputs.set(x)

    ## input spike train
    x_seq = jnp.asarray([[1., 1., 0., 0., 1.]], dtype=jnp.float32)
    ## desired output/epsp pulses
    y_seq = jnp.asarray([[0., 1., 0.998002, 0.996008, 1.9940181]], dtype=jnp.float32)

    outs = []
    reset_process.run()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
        clamp(x_t)
        advance_process.run(t=ts * 1., dt=dt)
        outs.append(a.epsp.get())
    outs = jnp.concatenate(outs, axis=1)
    #print(outs)

    ## output should equal input
    np.testing.assert_allclose(outs, y_seq, atol=1e-8)

test_expKernel1()
