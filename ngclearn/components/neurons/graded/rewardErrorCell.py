# %%

from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class RewardErrorCell(JaxComponent): ## Reward prediction error cell
    """
    A reward prediction error (RPE) cell.

    | --- Cell Input Compartments: ---
    | reward - current reward signal at time `t`
    | accum_reward - current accumulated episodic reward signal
    | --- Cell Output Compartments: ---
    | mu - current moving average prediction of reward at time `t`
    | rpe - current reward prediction error (RPE) signal
    | accum_reward - current accumulated episodic reward signal (IF online predictor not used)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        alpha: decay factor to apply to (exponential) moving average prediction

        ema_window_len: exponential moving average window length -- for use only
            in `evolve` step for updating episodic reward signals; (default: 10)

        use_online_predictor: use online prediction of reward signal (default: True)
            -- if set to False, then reward prediction will only occur upon a call
            to this cell's `evolve` function
    """
    def __init__(self, name, n_units, alpha, ema_window_len=10,
                 use_online_predictor=True, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## RPE meta-parameters
        self.alpha = alpha
        self.ema_window_len = ema_window_len
        self.use_online_predictor = use_online_predictor

        ## Layer Size Setup
        self.n_units = n_units
        self.batch_size = batch_size

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.mu = Compartment(restVals) ## reward predictor state(s)
        self.reward = Compartment(restVals) ## target reward signal(s)
        self.rpe = Compartment(restVals) ## reward prediction error(s)
        self.accum_reward = Compartment(restVals)  ## accumulated reward signal(s)
        self.n_ep_steps = Compartment(jnp.zeros((self.batch_size, 1))) ## number of episode steps taken

    @compilable
    def advance_state(self, dt):
        # Get the variables
        mu = self.mu.get()
        reward = self.reward.get()
        n_ep_steps = self.n_ep_steps.get()
        accum_reward = self.accum_reward.get()

        ## compute/update RPE and predictor values
        accum_reward = accum_reward + reward
        rpe = reward - mu
        if self.use_online_predictor:
            mu = mu * (1. - self.alpha) + reward * self.alpha
        n_ep_steps = n_ep_steps + 1

        # Update compartments
        self.mu.set(mu)
        self.rpe.set(rpe)
        self.n_ep_steps.set(n_ep_steps)
        self.accum_reward.set(accum_reward)

    @compilable
    def evolve(self, dt):
        # Get the variables
        mu = self.mu.get()
        n_ep_steps = self.n_ep_steps.get()
        accum_reward = self.accum_reward.get()

        if self.use_online_predictor:
            ## total episodic reward signal
            r = accum_reward/n_ep_steps
            mu = (1. - 1./self.ema_window_len) * mu + (1./self.ema_window_len) * r

        # Update compartment
        self.mu.set(mu)

    @compilable
    def reset(self): ## reset core components/statistics
        restVals = jnp.zeros((self.batch_size, self.n_units))
        mu = restVals
        rpe = restVals
        accum_reward = restVals
        n_ep_steps = jnp.zeros((self.batch_size, 1))
        self.mu.set(mu)
        self.rpe.set(rpe)
        self.accum_reward.set(accum_reward)
        self.n_ep_steps.set(n_ep_steps)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "RewardErrorCell - computes the reward prediction error "
                         "at each time step `t`; this is an online RPE estimator"
        }
        compartment_props = {
            "inputs":
                {"reward": "External reward signals/values"},
            "outputs":
                {"mu": "Current state of reward predictor",
                 "rpe": "Current value of reward prediction error at time `t`",
                 "accum_reward": "Current accumulated episodic reward signal (generally "
                                 "produced at the end of a control episode of `n_steps`)",
                 "n_ep_steps": "Number of episodic steps taken/tracked thus far "
                               "(since last `reset` call)",
                 "use_online_predictor": "Should an online reward predictor be used/maintained?"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "alpha": "Moving average decay factor",
            "ema_window_len": "Exponential moving average window length",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "rpe = reward - mu; mu = mu * (1 - alpha) + reward * alpha; "
                            "accum_reward = accum_reward + reward",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = RewardErrorCell("X", 9, 0.03)
    print(X)
