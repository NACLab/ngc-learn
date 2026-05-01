# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import LatencyCell
from numpy.testing import assert_array_equal
from ngclearn import MethodProcess, Context

def test_latencyCell1():
    name = "latency_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    T = 50  # 100 #5  ## number of simulation steps to run
    dt = 1.  # 0.1  # ms ## compute integration time constant
    tau = 1.
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = LatencyCell(
            "a", n_units=4, tau=tau, threshold=0.01, linearize=True,
            normalize=True, num_steps=T, clip_spikes=False
        )

        ## create and compile core simulation commands
        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)
        calc_spike_times_process = (MethodProcess("calc_sptimes_proc")
                                    >> a.calc_spike_times)
        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

        ## set up non-compiled utility commands
        def clamp(x):
            a.inputs.set(x)

    ## input spike train
    x_t = jnp.asarray([[0.02, 0.5, 1., 0.0]])

    targets = np.zeros((T, 4))
    targets[0, 2] = 1.
    targets[24, 1] = 1.
    targets[48, 0] = 1.
    targets[49, 3] = 1.
    targets = jnp.array(targets) ## gold-standard solution to check against

    outs = []
    reset_process.run()
    clamp(x_t)
    calc_spike_times_process.run()
    for ts in range(T):
        clamp(x_t)
        advance_process.run(t=ts * dt, dt=dt)
        ## naively extract simple statistics at time ts and print them to I/O
        s = a.outputs.get()
        outs.append(s)
        #print(" {}: s {} ".format(ts, jnp.squeeze(s)))
    outs = jnp.concatenate(outs, axis=0)

    ## output should equal input
    assert_array_equal(outs, targets)

#test_latencyCell1()
