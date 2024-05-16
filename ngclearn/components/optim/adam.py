# %%

from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def step_update(param, update, g1, g2, lr, beta1, beta2, time, eps):
    """
    Runs one step of Adam over a set of parameters given updates.
    The dynamics for any set of parameters is as follows:

    | g1 = beta1 * g1 + (1 - beta1) * update
    | g2 = beta2 * g2 + (1 - beta2) * (update)^2
    | g1_unbiased = g1 / (1 - beta1**time)
    | g2_unbiased = g2 / (1 - beta2**time)
    | param = param - lr * g1_unbiased / (sqrt(g2_unbiased) + epsilon)

    Args:
        param: parameter tensor to change/adjust

        update: update tensor to be applied to parameter tensor (must be same
            shape as "param")

        g1: first moment factor/correction factor to use in parameter update
            (must be same shape as "update")

        g2: second moment factor/correction factor to use in parameter update
            (must be same shape as "update")

        lr: global step size value to be applied to updates to parameters

        beta1: 1st moment control factor

        beta2: 2nd moment control factor

        time: current time t or iteration step/call to this Adam update

        eps: numberical stability coefficient (for calculating final update)

    Returns:
        adjusted parameter tensor (same shape as "param")
    """
    _g1 = beta1 * g1 + (1. - beta1) * update
    _g2 = beta2 * g2 + (1. - beta2) * jnp.square(update)
    g1_unb = _g1 / (1. - jnp.power(beta1, time))
    g2_unb = _g2 / (1. - jnp.power(beta2, time))
    _param = param - lr * g1_unb/(jnp.sqrt(g2_unb) + eps)
    return _param, _g1, _g2

class Adam(Component):
    """
    Implements the adaptive moment estimation (Adam) algorithm as a decoupled
    update rule given adjustments produced by a credit assignment algorithm/process.

    Args:
        learning_rate: step size coefficient for Adam update

        beta1: 1st moment control factor

        beta2: 2nd moment control factor

        epsilon: numberical stability coefficient (for calculating final update)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(name="adam")
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        # Compartment
        self.g1 = Compartment([0.0, 0.0]) # adam internal variable. Internal compartment. # [NOTE] VN: This needs to be revised
        self.g2 = Compartment([0.0, 0.0]) # adam internal variable. Internal compartment. # [NOTE] VN: This needs to be revised
        self.time_step = Compartment(0.0) # the time step. Internal compartment
        self.updates = Compartment(None) # the update or gradient. External compartment, to be wired
        self.theta = Compartment(None) # the current weight of other networks. External compartment, to be wired

    @staticmethod
    def pure_update(eta, beta1, beta2, eps, g1, g2, time_step, theta, updates):  ## apply adjustment to theta
        ## init statistics in a jitted way
        g1 = [jnp.zeros(theta[i].shape) * (1 - jnp.sign(time_step)) + g1[i] * jnp.sign(time_step) for i in range(len(theta))]
        g2 = [jnp.zeros(theta[i].shape) * (1 - jnp.sign(time_step)) + g2[i] * jnp.sign(time_step) for i in range(len(theta))]
        time_step = time_step + 1
        new_theta = []
        new_g1 = []
        new_g2 = []
        for i in range(len(theta)):
            px_i, g1_i, g2_i = step_update(theta[i], updates[i], g1[i],
                                           g2[i], eta, beta1,
                                           beta2, time_step, eps)
            new_theta.append(px_i)
            new_g1.append(g1_i)
            new_g2.append(g2_i)
        return new_g1, new_g2, time_step, new_theta

    @resolver(pure_update, output_compartments=["g1", "g2", "time_step", "theta"])
    def update(self, g1, g2, time_step, theta):
        self.g1.set(g1)
        self.g2.set(g2)
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
                component.update()

    with Context("Bar") as bar:
        a1 = RateCell("a1", 2, 0.01)
        a2 = RateCell("a2", 2, 0.01)
        adam = Adam()
        a2.j << a1.zF
        adam.theta << a1.zF
        adam.updates << a2.zF
        cmd = AdvanceCommand(components=[a1, a2], command_name="Advance")
        update_cmd = UpdateCommand(components=[adam], command_name="Update")

    compiled_cmd, arg_order = cmd.compile()
    wrapped_cmd = wrapper(jit(compiled_cmd))

    compiled_update_cmd, _ = update_cmd.compile()
    wrapped_update_cmd = wrapper(jit(compiled_update_cmd))

    dt = 0.01
    for t in range(3):
        a1.j.set(jnp.asarray([[2.8, 9.3]]))
        wrapped_cmd(t, dt)
        wrapped_update_cmd()
        print(f"Step {t} - [a1] j: {a1.j.value}, j_td: {a1.j_td.value}, z: {a1.z.value}, zF: {a1.zF.value}")
        print(f"Step {t} - [a2] j: {a2.j.value}, j_td: {a2.j_td.value}, z: {a2.z.value}, zF: {a2.zF.value}")
        print(f"Step {t} - [adam] g1: {adam.g1.value}, g2: {adam.g2.value}, theta: {adam.theta.value}, updates: {adam.updates.value}, time_step: {adam.time_step.value}")

