# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import BernoulliCell
#from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngclearn import MethodProcess, Context


def test_bernoulliCell1():
    name = "bernoulli_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    #T = 300  # ms
    # ---- build a simple Bernoulli cell system ----
    with Context(name) as ctx:
        a = BernoulliCell(name="a", n_units=1, key=subkeys[0])

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

        ## set up non-compiled utility commands
        def clamp(x):
            a.inputs.set(x)

    ## input spike train
    x_seq = jnp.asarray([[1., 1., 0., 0., 1.]], dtype=jnp.float32)

    outs = []
    reset_process.run()
    for ts in range(x_seq.shape[1]):
        x_t = jnp.array([[x_seq[0,ts]]]) ## get data at time t
        clamp(x_t)
        advance_process.run(t=ts*1., dt=dt)
        outs.append(a.outputs.get())
    outs = jnp.concatenate(outs, axis=1)

    ## output should equal input
    assert_array_equal(outs, x_seq)
    #print(outs)

#test_bernoulliCell1()
