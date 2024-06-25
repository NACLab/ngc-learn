from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngcsimlib.logger import info
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

@jit
def _dfv_internal(inputs, Elg, tau_elg, elg_decay=1., input_scale=1.): ## raw eligbility dynamics
    dElg_dt = -Elg * elg_decay + inputs * input_scale
    dElg_dt = dElg_dt * (1./tau_elg)
    return dElg_dt

def _dfElg(t, Elg, params): ## eligbility trace dynamics wrapper
    inputs, tau_elg, elg_decay, input_scale = params
    dElg_dt = _dfv_internal(inputs, Elg, tau_elg, elg_decay, input_scale)
    return dElg_dt

class EligibilityTrace(JaxComponent): ## eligibility trace
    """
    A generic eligibility trace construct for tracking dynamics of a tensor
    object.

    | --- Cell/Op Input Compartments: ---
    | inputs - input (takes in external signals/tensor)
    | modulator - scalar (neuro)modulatory signal (takes in external signals)
    | --- Cell/Op Output Compartments: ---
    | eligibility - current eligibility trace at time `t`
    | modded_outputs - neuro-modulated eligibility output signals

    Args:
        name: the string name of this trace

        shape: tuple specifying shape of this trace cable (note that this tuple can
            specify N-D tensor shapes)

        tau_elg: eligibility time constant

        elg_decay: eligiblity decacy magnitude/constant; (must be non-zero,
            default 1.)

        input_scale: input (at time `t`) scaling factor (default: 1.)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration)
            (Default: "euler")
    """

    # Define Functions
    def __init__(self, name, shape, tau_elg=100., elg_decay=1., input_scale=1.,
                 integration_type="euler", batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size ## Note: batch size technically means nothing to an eligibility trace

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Eligibility meta-parameters
        self.shape = shape
        self.tau_elg = tau_elg ## eligiblity time constant
        self.elg_decay = elg_decay ## trace decay time constant
        self.input_scale = input_scale

        ## Set up eligibility trace compartment values
        initVals = jnp.zeros(shape)
        self.inputs = Compartment(initVals)  ## input that this trace tracks
        self.modulator = Compartment(jnp.ones((1, 1)))
        self.eligibility = Compartment(initVals)  ## eligibility trace condition matrix
        self.modded_outputs = Compartment(initVals)

    @staticmethod
    def _advance_state(t, dt, intgFlag, input_scale, tau_elg, elg_decay,
                       inputs, eligibility):
        elg_params = (inputs, tau_elg, elg_decay, input_scale)
        if intgFlag == 1:
            _, eligibility = step_rk2(0., eligibility, _dfElg, dt, elg_params)
        else:
            _, eligibility = step_euler(0., eligibility, _dfElg, dt, elg_params)
        return eligibility

    @resolver(_advance_state)
    def advance_state(self, eligibility):
        self.eligibility.set(eligibility)

    @staticmethod
    def _evolve(t, dt, eligibility, modulator):
        return eligibility * modulator

    @resolver(_evolve)
    def evolve(self, modded_outputs):
        self.modded_outputs.set(modded_outputs)

    @staticmethod
    def _reset(shape):
        initVals = jnp.zeros(shape)
        return initVals, initVals, initVals

    @resolver(_reset)
    def reset(self, eligibility, inputs, modded_outputs):
        self.eligibility.set(eligibility)
        self.inputs.set(inputs)
        self.modded_outputs.set(modded_outputs)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, eligibility=self.eligibility.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.eligibility.set(data['eligibility'])

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "EligibilityTrace - maintains a set/tensor of eligibility "
                         "trace values over time"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "modulator": "Takes in external modulatory signal values"},
            "states":
                {"key": "JAX PRNG key"},
            "outputs":
                {"eligibility": "Current state of eligibility trace at time `t` (Elg)",
                 "modded_outputs": "Current eligibility scaled by external "
                                   "(neuro)modulatory signal (mod)"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "tau_elg": "Eligibility time constant",
            "tau_d": "Eligibility decay magnitude/constant",
            "input_scale": "Input signal scaling factor",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "modded_outputs = Elg * mod; "
                            "dElg/dt = -Elg * elg_decay + inputs * input_scale",
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
