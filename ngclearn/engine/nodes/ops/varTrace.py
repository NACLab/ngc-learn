from ngclearn.engine.nodes.ops.op import Op
from jax import random, numpy as jnp, jit
from functools import partial
#from functools import partial

@partial(jit, static_argnums=[5,7])
def run_varfilter(t, x, x_tr, dt, tau_tr, incr_pos, a_delta, decay_type="lin"):
    """
    variable trace filter dynamics
    """
    _x_tr = None
    if "exp" in decay_type: ## apply exponential decay
        gamma = jnp.exp(-dt/tau_tr)
        _x_tr = x_tr * gamma
    elif "lin" in decay_type: ## default => apply linear "lin" decay
        _x_tr = x_tr + (-x_tr) * (dt / tau_tr)
    elif "step" in decay_type:
        _x_tr = x_tr * 0
    else:
        print("ERROR: decay.type = {} unrecognized".format(decay_type))
        sys.exit(1)
    if incr_pos == True: ## perform additive form of trace ODE
        _x_tr = _x_tr + x * a_delta
        #_x_tr = x_tr + (-x_tr) * (dt / tau_tr) + x * a_delta
    else: ## run piecewise ODE variant of trace
        _x_tr = _x_tr * (1. - x) + x
        #_x_tr = ( x_tr + (-x_tr) * (dt / tau_tr) ) * (1. - x) + x
    return _x_tr

## Variable trace function node
class VarTrace(Op):  # inherits from Node class
    """
    A variable trace (filter) functional node.

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        tau_tr: trace time constant

        incr_pos: if True, increment ODE by this value, else set ODE to 1 upon increase

        a_delta: increment trace +1 instead of set to 1 if spike (DEFAULT: 1)

        decay_type: decay type applied to ODE integration; low-pass filter configuration

            :Note: values this can be are;
                "lin" = linear trace filter
                "exp" = exponential trace filter
                "step" = step trace filter

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, tau_tr=50., incr_pos=False,
                 a_delta=1., decay_type="lin", key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.tau_tr = tau_tr  ## trace time constant
        self.incr_pos = incr_pos ## increment trace +1 instead of set to 1 if spike
        self.a_delta = a_delta ## if incr_pos == True, increase ODE by this value
        self.decay_type = decay_type ## decay type applied to ODE integration

        # cell compartments
        self.comp["in"] = None
        self.comp["z"] = None ## resultant variable trace
        self.comp["s_prev"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        s = self.comp["in"] ## get incoming spike input readout
        z = self.comp["z"]
        if self.decay_type == "step":
            _z = self.comp["s_prev"] + 0 # trace(t) = s(t-1)
            self.comp["s_prev"] = s + 0 # s(t-1) <= s(t)
        else:
            _z = run_varfilter(self.t, s, z, self.dt, self.tau_tr,
                               incr_pos=self.incr_pos, a_delta=self.a_delta,
                               decay_type=self.decay_type)  ## run filter
        self.comp["z"] = _z

    def custom_dump(self, node_directory, template=False):
        required_keys = ['tau_tr', 'incr_pos', 'a_delta', 'decay_type']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    @staticmethod
    def get_default_out():
        """
        Returns the value within compartment ``z``
        """
        return 'z'

    comp_z = "z"
    comp_s_prev = "s_prev"

class_name = VarTrace.__name__
