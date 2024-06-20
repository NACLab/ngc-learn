from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngcsimlib.logger import info
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

@jit
def _dfv_internal(inputs, Elg, tau_elg, elg_decay=1.): ## raw eligbility dynamics
    dElg_dt = -Elg * elg_decay + inputs
    dElg_dt = dElg_dt * (1./tau_elg)
    return dElg_dt

def _dfElg(t, Elg, params): ## eligbility trace dynamics wrapper
    inputs, tau_elg, elg_decay = params
    dElg_dt = _dfv_internal(inputs, Elg, tau_elg, elg_decay)
    return dElg_dt

class EligibilityTrace(JaxComponent): ## eligibility trace
    """
    A generic eligibility trace construct. Note that an eligibility trace can
    be viewed as a dynamic parameter tensor.

    | --- Cell/Op Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell/Op Output Compartments: ---
    | Elg - current eligibility trace at time `t`

    Args:
        name: the string name of this trace

        shape: tuple specifying shape of this trace cable (note tuple can
            specify N-D tensor shapes)

        tau_elg: eligibility time constant

        elg_decay: eligiblity decacy magnitude/constant; (must be non-zero,
            default 1.)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration)
            (Default: "euler")
    """

    # Define Functions
    def __init__(self, name, shape, tau_elg=100., elg_decay=1.,
                 integration_type="euler", **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = 1 ## Note: batch size technically means nothing to an eligibility trace

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Eligibility meta-parameters
        self.shape = shape
        self.tau_elg = tau_elg ## eligiblity time constant
        self.elg_decay = elg_decay ## trace decay time constant

        ## Set up eligibility trace compartment values
        initVals = jnp.zeros(shape)
        self.inputs = Compartment(initVals)  ## input that this trace tracks
        self.Elg = Compartment(initVals)  ## eligibility trace condition matrix

    @staticmethod
    def _advance_state(t, dt, intgFlag, tau_elg, elg_decay, inputs, Elg):
        elg_params = (inputs, tau_elg, elg_decay)
        if intgFlag == 1:
            _, Elg = step_rk2(0., Elg, _dfElg, dt, elg_params)
        else:
            _, Elg = step_euler(0., Elg, _dfElg, dt, elg_params)
        return Elg

    @resolver(_advance_state)
    def advance_state(self, Elg):
        self.Elg.set(Elg)

    @staticmethod
    def _reset(shape):
        initVals = jnp.zeros(shape)
        return initVals, initVals

    @resolver(_reset)
    def reset(self, Elg, inputs):
        self.Elg.set(Elg)
        self.inputs.set(inputs)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, Elg=self.Elg.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.Elg.set(data['Elg'])

    def help(self): ## component help function
        properties = {
            "cell_type": "EligibilityTrace - maintains a set/tensor of eligibility "
                         "trace values over time"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX RNG key"},
            "outputs_compartments":
                {"Elg": "Current state of eligibility trace at time `t`"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "tau_elg": "Eligibility time constant",
            "tau_d": "Eligibility decay magnitude/constant"
        }
        info = {self.name: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b; "
                            "dW/dt = W_full * u * x * inputs",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = EligibilityTrace("Wab", (2, 3))
    print(Wab)
