from jax import random, numpy as jnp, jit
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
import jax
import jax.numpy as jnp
import numpy as np

from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import create_function

def gaussian_logpdf(event, mean, stddev):
    # scale_sqrd = stddev ** 2
    # log_normalizer = jnp.log(2 * jnp.pi * scale_sqrd)
    # quadratic = (jax.lax.stop_gradient(event - 2 * mean) + mean)**2 / scale_sqrd
    # return - 0.5 * (log_normalizer + quadratic)
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(stddev) - 0.5 * (  (jax.lax.stop_gradient(event - 2 * mean) + mean) / stddev  )**2

class REINFORCESynapse(DenseSynapse):

    # Define Functions
    def __init__(
            self, name, shape, eta=1e-4, decay=0.99, weight_init=None, resist_scale=1., act_fx=None,
            p_conn=1., w_bound=1., batch_size=1, seed=None, **kwargs
    ):
        # This is because we have weights mu and weight log sigma
        input_dim, output_dim = shape
        super().__init__(name, (input_dim, output_dim * 2), weight_init, None, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## Synaptic hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound #1. ## soft weight constraint
        self.eta = eta ## learning rate

        ## Compartment setup
        self.dWeights = Compartment(self.weights.value * 0)
        # self.eta = Compartment(jnp.ones((1, 1)) * eta) ## global learning rate # For eligiblity traces later
        self.objective = Compartment(jnp.zeros(()))
        self.outputs = Compartment(jnp.zeros((batch_size, output_dim)))
        self.rewards = Compartment(jnp.zeros((batch_size,))) # the normalized reward (r - r_hat), input compartment
        self.act_fx, self.dact_fx = create_function(act_fx if act_fx is not None else "identity")
        # self.seed = Component(seed)
        self.accumulated_gradients = Compartment(jnp.zeros((input_dim, output_dim * 2)))
        self.decay = decay
        self.step_count = Compartment(jnp.zeros(()))
        self.learning_mask = Compartment(jnp.zeros(()))
        # self.seed = Component(jnp.array(seed) if seed is not None else jnp.array(42, dtype=jnp.int32))
        self.seed = Compartment(jax.random.PRNGKey(seed if seed is not None else 42))

    @staticmethod
    def _compute_update(dt, inputs, rewards, act_fx, weights, seed):
        W_mu, W_logstd = jnp.split(weights, 2, axis=-1) # (input_dim, output_dim * 2) => (input_dim, output_dim), (input_dim, output_dim)
        # Forward pass
        activation = act_fx(inputs)
        mean = activation @ W_mu
        logstd = activation @ W_logstd
        clip_logstd = jnp.clip(logstd, -10.0, 2.0)
        std = jnp.exp(clip_logstd)
        # Sample using reparameterization trick
        epsilon = jax.random.normal(seed, mean.shape)
        sample = epsilon * std + mean
        outputs = sample # the actual action that we take
        # Compute log probability density of the Gaussian
        log_prob = gaussian_logpdf(sample, mean, std).sum(-1)
        # Compute objective (negative REINFORCE objective)
        objective = (-log_prob * rewards).mean() * 1e-2

        # Backward pass
        batch_size = inputs.shape[0] # B
        dL_dlogp = -rewards[:, None] * 1e-2 / batch_size # (B, 1)

        # Compute gradients manually based on the derivation
        # dL/dmu = -(r-r_hat) * dlog_prob/dmu = -(r-r_hat) * -(sample-mu)/sigma^2
        # -(sample - mean) instead of (sample - mean) because we are doing straight-through gradient in the log_prob function
        # therefore, computation including the mean in such function does not contribute to the gradient
        dlog_prob_dmean = -(sample - mean) / (std ** 2)
        dL_dmean = dL_dlogp * dlog_prob_dmean # (B, A)
        dL_dWmu = activation.T @ dL_dmean

        # dL/dlog(sigma) = -(r-r_hat) * dlog_prob/dlog(sigma) = -(r-r_hat) * (((sample-mu)/sigma)^2 - 1)
        dlog_prob_dlogstd = (sample - mean)**2 / std**3 - 1.0 / std
        dL_dstd = dL_dlogp * dlog_prob_dlogstd
        # Apply gradient clipping for logstd
        dL_dlogstd = jnp.where(
            (logstd <= -10.0) | (logstd >= 2.0),
            0.0,  # Zero gradient when clipped
            dL_dstd * std
        )
        dL_dWlogstd = activation.T @ dL_dlogstd # (I, B) @ (B, A) = (I, A)

        # Update weights, negate the gradient because gradient ascent in ngc-learn
        dW = jnp.concatenate([-dL_dWmu, -dL_dWlogstd], axis=-1)
        # Finally, return metrics if needed
        return dW, objective, outputs

    @transition(output_compartments=["weights", "dWeights", "objective", "outputs", "accumulated_gradients", "step_count", "seed"])
    @staticmethod
    def evolve(dt, w_bound, inputs, rewards, act_fx, weights, eta, learning_mask, decay, accumulated_gradients, step_count, seed):
        main_seed, sub_seed = jax.random.split(seed)
        dWeights, objective, outputs = REINFORCESynapse._compute_update(
            dt, inputs, rewards, act_fx, weights, sub_seed
        )
        ## do a gradient ascent update/shift
        weights = (weights + dWeights * eta) * learning_mask + weights * (1.0 - learning_mask) # update the weights only where learning_mask is 1.0
        ## enforce non-negativity
        eps = 0.0 # 0.01 # 0.001
        weights = jnp.clip(weights, eps, w_bound - eps)  # jnp.abs(w_bound))
        step_count += 1
        accumulated_gradients = (step_count - 1) / step_count * accumulated_gradients * decay + 1.0 / step_count * dWeights # EMA update of accumulated gradients
        step_count = step_count * (1 - learning_mask) # reset the step count to 0 when we have learned
        return weights, dWeights, objective, outputs, accumulated_gradients, step_count, main_seed

    @transition(output_compartments=["inputs", "outputs", "objective", "rewards", "dWeights", "accumulated_gradients", "step_count", "seed"])
    @staticmethod
    def reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        objective = jnp.zeros(())
        rewards = jnp.zeros((batch_size,))
        dWeights = jnp.zeros(shape)
        accumulated_gradients = jnp.zeros((shape[0], shape[1] * 2))
        step_count = jnp.zeros(())
        seed = jax.random.PRNGKey(42)
        return inputs, outputs, objective, rewards, dWeights, accumulated_gradients, step_count, seed

    @classmethod
    def help(cls): ## component help function
        properties = {

        }
        compartment_props = {

        }
        hyperparams = {

        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                # "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                #             "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
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
