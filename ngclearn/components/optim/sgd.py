# %%

from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def step_update(param, update, lr):
    """
    Runs one step of SGD over a set of parameters given updates.

    Args:
        lr: global step size to apply when adjusting parameters

    Returns:
        adjusted parameter tensor (same shape as "param")
    """
    _param = param - lr * update
    return _param

class SGD(Component):
    """
    Implements stochastic gradient descent (SGD) as a decoupled update rule
    given adjustments produced by a credit assignment algorithm/process.

    Args:
        learning_rate: step size coefficient for SGD update
    """
    def __init__(self, learning_rate=0.001):
        super().__init__(name="sgd")
        self.eta = learning_rate

        # Compartment
        self.time_step = Compartment(0.0) # the time step. Internal compartment
        self.updates = Compartment(None) # the update or gradient. External compartment, to be wired
        self.theta = Compartment(None) # the current weight of other networks. External compartment, to be wired

    @staticmethod
    def pure_update(t, dt, eta, time_step, theta, updates): ## apply adjustment to theta
        time_step = time_step + 1
        new_theta = []
        for i in range(len(theta)):
            px_i = step_update(theta[i], updates[i], eta)
            new_theta.append(px_i)
        return time_step, new_theta

    @resolver(pure_update, output_compartments=["time_step", "theta"])
    def update(self, time_step, theta):
        self.time_step.set(time_step)
        self.theta.set(theta)

if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command
    from ngclearn.components.neurons.graded.rateCell import RateCell

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

    class UpdateCommand(Command):
        compile_key = "update"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.update(t=t, dt=dt)

    with Context("Bar") as bar:
        a1 = RateCell("a1", 2, 0.01)
        a2 = RateCell("a2", 2, 0.01)
        sgd = SGD()
        a2.j << a1.zF
        sgd.theta << a1.zF
        sgd.updates << a2.zF
        cmd = AdvanceCommand(components=[a1, a2], command_name="Advance")
        update_cmd = UpdateCommand(components=[sgd], command_name="Update")

    compiled_cmd, arg_order = cmd.compile(loops=1, param_generator=lambda loop_id: [loop_id + 1, 0.1])
    wrapped_cmd = wrapper(compiled_cmd)

    compiled_update_cmd, _ = update_cmd.compile(loops=1, param_generator=lambda loop_id: [loop_id + 1, 0.1])
    wrapped_update_cmd = wrapper(compiled_update_cmd)

    for i in range(3):
        a1.j.set(10)
        wrapped_cmd()
        wrapped_update_cmd()
        print(f"Step {i} - [a1] j: {a1.j.value}, j_td: {a1.j_td.value}, z: {a1.z.value}, zF: {a1.zF.value}")
        print(f"Step {i} - [a2] j: {a2.j.value}, j_td: {a2.j_td.value}, z: {a2.z.value}, zF: {a2.zF.value}")
        print(f"Step {i} - [sgd] theta: {sgd.theta.value}, updates: {sgd.updates.value}, time_step: {sgd.time_step.value}")
    a1.reset()
    a2.reset()
    print(f"Reset: [a1] j: {a1.j.value}, j_td: {a1.j_td.value}, z: {a1.z.value}, zF: {a1.zF.value}")
    print(f"Reset: [a2] j: {a2.j.value}, j_td: {a2.j_td.value}, z: {a2.z.value}, zF: {a2.zF.value}")

