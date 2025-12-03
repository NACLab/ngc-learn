# %%

from jax import random, numpy as jnp, jit
from ngclearn import compilable, Compartment

from ngclearn.utils.model_utils import clip, d_clip
import jax
#import numpy as np

from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import create_function

def _gaussian_logpdf(event, mean, stddev):
  scale_sqrd = stddev ** 2
  log_normalizer = jnp.log(2 * jnp.pi * scale_sqrd)
  quadratic = (event - mean)**2 / scale_sqrd
  return - 0.5 * (log_normalizer + quadratic)


def _compute_update(
        dt, inputs, rewards, act_fx, weights, seed, mu_act_fx, dmu_act_fx, mu_out_min, mu_out_max, scalar_stddev
):
    learning_stddev_mask = jnp.asarray(scalar_stddev <= 0.0, dtype=jnp.float32)
    # (input_dim, output_dim * 2) => (input_dim, output_dim), (input_dim, output_dim)
    W_mu, W_logstd = jnp.split(weights, 2, axis=-1)
    # Forward pass
    activation = act_fx(inputs)
    mean = activation @ W_mu
    fx_mean = mu_act_fx(mean)
    logstd = activation @ W_logstd
    clip_logstd = clip(logstd, -10.0, 2.0)
    std = jnp.exp(clip_logstd)
    std = learning_stddev_mask * std + (1.0 - learning_stddev_mask) * scalar_stddev # masking trick
    # Sample using reparameterization trick
    epsilon = jax.random.normal(seed, fx_mean.shape)
    sample = epsilon * std + fx_mean
    sample = jnp.clip(sample, mu_out_min, mu_out_max)
    outputs = sample # the actual action that we take
    # Compute log probability density of the Gaussian
    log_prob = _gaussian_logpdf(sample, fx_mean, std).sum(-1)
    # Compute objective (negative REINFORCE objective)
    objective = (-log_prob * rewards).mean() * 1e-2

    # Backward pass
    batch_size = inputs.shape[0] # B
    dL_dlogp = -rewards[:, None] * 1e-2 / batch_size # (B, 1)

    # Compute gradients manually based on the derivation
    # dL/dmu = -(r-r_hat) * dlog_prob/dmu = -(r-r_hat) * -(sample-mu)/sigma^2
    dlog_prob_dfxmean = (sample - fx_mean) / (std ** 2)
    dL_dmean = dL_dlogp * dlog_prob_dfxmean * dmu_act_fx(mean) # (B, A)
    dL_dWmu = activation.T @ dL_dmean

    # dL/dlog(sigma) = -(r-r_hat) * dlog_prob/dlog(sigma) = -(r-r_hat) * (((sample-mu)/sigma)^2 - 1)
    dlog_prob_dlogstd = - 1.0 / std + (sample - fx_mean)**2 / std**3
    dL_dstd = dL_dlogp * dlog_prob_dlogstd
    # Apply gradient clipping for logstd
    dL_dlogstd = d_clip(logstd, -10.0, 2.0) * dL_dstd * std
    dL_dWlogstd = activation.T @ dL_dlogstd # (I, B) @ (B, A) = (I, A)
    dL_dWlogstd = dL_dWlogstd * learning_stddev_mask # there is no learning for the scalar stddev

    # Update weights, negate the gradient because gradient ascent in ngc-learn
    dW = jnp.concatenate([-dL_dWmu, -dL_dWlogstd], axis=-1)
    # Finally, return metrics if needed
    return dW, objective, outputs


class REINFORCESynapse(DenseSynapse):
    """
    A stochastic synapse implementing the REINFORCE algorithm (policy gradient method). This synapse
    uses Gaussian distributions for generating actions and performs gradient-based updates.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (sampled actions from Gaussian distribution)
    | weights - current value matrix of synaptic efficacies (contains both mean and log-std parameters)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | rewards - reward signals used to modulate weight updates (takes in external signals)
    | objective - scalar value of the current loss/objective
    | accumulated_gradients - exponential moving average of gradients for tracking learning progress
    | step_count - counter for number of learning steps
    | learning_mask - binary mask determining when learning occurs
    | seed - JAX PRNG key for random sampling

    Args:
        name: the string name of this component

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: learning rate for weight updates (Default: 1e-4)

        decay: decay factor for computing exponential moving average of gradients (Default: 0.99)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.)

        act_fx: activation function to apply to inputs (Default: "identity")

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure

        w_bound: upper bound for weight clipping (Default: 1.)

        batch_size: batch size dimension of this component (Default: 1)

        seed: random seed for reproducibility (Default: 42)

        mu_act_fx: activation function to apply to the mean of the Gaussian distribution (Default: "identity")
    """

    # Define Functions
    def __init__(
            self, name, shape, eta=1e-4, decay=0.99, weight_init=None, resist_scale=1., act_fx=None,
            p_conn=1., w_bound=1., batch_size=1, seed=None, mu_act_fx=None, mu_out_min=-jnp.inf, mu_out_max=jnp.inf,
            scalar_stddev=-1.0, **kwargs
    ) -> None:
        # This is because we have weights mu and weight log sigma
        input_dim, output_dim = shape
        super().__init__(
            name, (input_dim, output_dim * 2), weight_init, None, resist_scale, p_conn,
            batch_size=batch_size, **kwargs
        )

        ## Synaptic hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound #1. ## soft weight constraint
        self.eta = eta ## learning rate
        # self.out_min = out_min
        # self.out_max = out_max
        self.mu_act_fx, self.dmu_act_fx = create_function(mu_act_fx if mu_act_fx is not None else "identity")
        self.mu_out_min = mu_out_min
        self.mu_out_max = mu_out_max
        self.scalar_stddev = scalar_stddev

        ## Compartment setup
        self.dWeights = Compartment(self.weights.get() * 0)
        # self.eta = Compartment(jnp.ones((1, 1)) * eta) ## global learning rate # For eligiblity traces later
        self.objective = Compartment(jnp.zeros(()))
        self.outputs = Compartment(jnp.zeros((batch_size, output_dim)))
        self.rewards = Compartment(jnp.zeros((batch_size,))) # the normalized reward (r - r_hat), input compartment
        self.act_fx, self.dact_fx = create_function(act_fx if act_fx is not None else "identity")
        self.accumulated_gradients = Compartment(jnp.zeros((input_dim, output_dim * 2)))
        self.decay = decay
        self.step_count = Compartment(jnp.zeros(()))
        self.learning_mask = Compartment(jnp.zeros(()))
        self.seed = Compartment(jax.random.PRNGKey(seed if seed is not None else 42))

    @compilable
    def evolve(self, dt):
        # Get compartment values
        weights = self.weights.get()
        dWeights = self.dWeights.get()
        objective = self.objective.get()
        outputs = self.outputs.get()
        accumulated_gradients = self.accumulated_gradients.get()
        step_count = self.step_count.get()
        seed = self.seed.get()
        inputs = self.inputs.get()
        rewards = self.rewards.get()

        # Main logic
        main_seed, sub_seed = jax.random.split(seed)
        dWeights, objective, outputs = _compute_update(
            dt, inputs, rewards, self.act_fx, weights, sub_seed, self.mu_act_fx, self.dmu_act_fx, self.mu_out_min, self.mu_out_max, self.scalar_stddev
        )
        ## do a gradient ascent update/shift
        weights = (weights + dWeights * self.eta) * self.learning_mask + weights * (1.0 - self.learning_mask.get()) # update the weights only where learning_mask is 1.0
        ## enforce non-negativity
        eps = 0.0 # 0.01 # 0.001
        weights = jnp.clip(weights, eps, self.w_bound - eps)  # jnp.abs(w_bound))
        step_count += 1
        accumulated_gradients = (step_count - 1) / step_count * accumulated_gradients * self.decay + 1.0 / step_count * dWeights # EMA update of accumulated gradients
        step_count = step_count * (1 - self.learning_mask.get()) # reset the step count to 0 when we have learned

        # Set updated compartment values
        self.weights.set(weights)
        self.dWeights.set(dWeights)
        self.objective.set(objective)
        self.outputs.set(outputs)
        self.accumulated_gradients.set(accumulated_gradients)
        self.step_count.set(step_count)
        self.seed.set(main_seed)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        inputs = preVals
        outputs = postVals
        objective = jnp.zeros(())
        rewards = jnp.zeros((self.batch_size,))
        dWeights = jnp.zeros(self.shape)
        accumulated_gradients = jnp.zeros((self.shape[0], self.shape[1] * 2))
        step_count = jnp.zeros(())
        seed = jax.random.PRNGKey(42)

        hasattr(self.inputs, 'targeted') and not self.inputs.targeted and self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.objective.set(objective)
        self.rewards.set(rewards)
        self.dWeights.set(dWeights)
        self.accumulated_gradients.set(accumulated_gradients)
        self.step_count.set(step_count)
        self.seed.set(seed)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "REINFORCESynapse - implements a stochastic synaptic cable that uses "
                            "the REINFORCE algorithm (policy gradient) to update weights based on rewards"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "rewards": "Takes in reward signals for modulating weight updates. The reward is often normalized by baseline reward (r - r_hat)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values (mean and log-std)",
                 "dWeights": "Weight update values",
                 "accumulated_gradients": "EMA of gradients over time",
                 "step_count": "Counter for learning steps",
                 "learning_mask": "Binary mask determining when learning occurs",
                 "seed": "a single integer as initial jax PRNG key for this component"},
            "outputs":
                {"outputs": "Output samples from Gaussian distribution",
                 "objective": "Current value of the loss/objective function"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "eta": "Learning rate for weight updates",
            "decay": "Decay factor for EMA of gradients",
            "weight_init": "Initialization conditions for synaptic weight values",
            "resist_scale": "Resistance level scaling factor applied to output",
            "act_fx": "Activation function to apply to inputs",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "w_bound": "Upper bound for weight clipping",
            "batch_size": "Batch size dimension of this component",
            "seed": "Random seed for reproducibility",
            "mu_act_fx": "Activation function to apply to the mean of the Gaussian distribution"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "mean = act_fx(inputs) @ W_mu; fx_mean = mu_act_fx(mean); logstd = act_fx(inputs) @ W_logstd; "
                            "outputs ~ N(fx_mean, exp(logstd)); "
                            "dW = -grad_reinforce(rewards, log_prob(outputs)). ",
                            "Check compute_update() for more details."
                "hyperparameters": hyperparams}
        return info


if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        syn = REINFORCESynapse(
            name="reinforce_syn",
            shape=(3, 2)
        )
        # Wab = syn.weights.get()
    print(syn)

