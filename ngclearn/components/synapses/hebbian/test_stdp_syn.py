from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params, normalize_matrix
import time
from ngcsimlib.compartment import All_compartments
from ngcsimlib.context import Context
from ngcsimlib.commands import Command
from ngclearn.components.synapses.hebbian.traceSTDPSynapse import TraceSTDPSynapse
from ngclearn.components.input_encoders.poissonCell import PoissonCell

def wrapper(compiled_fn):
    def _wrapped(*args):
        # vals = jax.jit(compiled_fn)(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
        vals = compiled_fn(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
        for key, value in vals.items():
            All_compartments[str(key)].set(value)
        return vals
    return _wrapped

class AdvanceCommand(Command):
    compile_key = "advance"
    def __call__(self, t=None, dt=None, *args, **kwargs):
        for component in self.components:
            component.gather()
            component.advance(t=t, dt=dt)

class EvolveCommand(Command):
    compile_key = "evolve"
    def __call__(self, t=None, dt=None, *args, **kwargs):
        for component in self.components:
            component.evolve(t=t, dt=dt)

class ResetCommand(Command):
    compile_key = "reset"
    def __call__(self, t=None, dt=None, *args, **kwargs):
        for component in self.components:
            component.reset(t=t, dt=dt)

dkey = random.PRNGKey(1234)
with Context("Context") as context:
    z0 = PoissonCell("z0", n_units=1, max_freq=128., key=dkey)
    W = TraceSTDPSynapse("W", shape=(1,2), eta=0.1, Aplus=1., Aminus=0.9, mu=0.,
                         preTrace_target=0.0, wInit=("uniform", 0.025, 0.8),
                         w_norm=None, norm_T=250, key=dkey) #78.5, norm_T=250)
    z1 = PoissonCell("z1", n_units=2, max_freq=128., key=dkey)
    W.inputs << z0.outputs
    z1.inputs << W.outputs
    W.postSpike << z1.outputs
    advance_cmd = AdvanceCommand(components=[W], command_name="Advance")
    evolve_cmd = EvolveCommand(components=[W], command_name="Evolve")
    reset_cmd = ResetCommand(components=[W], command_name="Reset")

T = 30 #250
dt = 1.

compiled_advance_cmd, _ = advance_cmd.compile()
wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

compiled_evolve_cmd, _ = evolve_cmd.compile()
wrapped_evolve_cmd = wrapper(jit(compiled_evolve_cmd))

compiled_reset_cmd, _ = reset_cmd.compile()
wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

t = 0.
for i in range(T): # i is "t"
    val = ((i % 2 == 0)) * 1.
    pre_spk = jnp.asarray([[val]])
    pre_tr = jnp.asarray([[val]])
    post_spk = jnp.expand_dims(jnp.asarray([val, 1. - val]), axis=0)
    post_tr = post_spk + 0.
    z0.inputs.set(pre_spk)
    #W.inputs.set(pre_spk)
    wrapped_advance_cmd(t, dt) ## pass in t and dt and run step forward of simulation
    #W.preSpike.set(pre_spk)
    #W.preTrace.set(pre_tr)
    #W.postSpike.set(post_spk)
    #W.postTrace.set(post_tr)
    wrapped_evolve_cmd(t, dt) ## pass in t and dt and run step forward of simulation
    t = t + dt
    print(f"---[ Step {i} ]---")
    print(f"[W] in: {W.inputs.value}, out: {W.outputs.value}, preS: {W.preSpike.value}, " \
          f"preTr: {W.preTrace.value}, postS: {W.postSpike.value}, postTr: {W.postTrace.value}," \
          f"W: {W.weights.value}")
#a.reset()
wrapped_reset_cmd()
print(f"---[ After reset ]---")
print(f"[W] in: {W.inputs.value}, out: {W.outputs.value}, preS: {W.preSpike.value}, " \
      f"preTr: {W.preTrace.value}, postS: {W.postSpike.value}, postTr: {W.postTrace.value}," \
      f"W: {W.weights.value}")
