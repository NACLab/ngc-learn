from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats

class RewardErrorCell(JaxComponent): ## Reward prediction error cell
    """
    A reward prediction error (RPE) cell.

    | --- Cell Input Compartments: ---
    | reward - current reward signal at time `t`
    | --- Cell Output Compartments: ---
    | mu - current moving average prediction of reward at time `t`

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        alpha: decay factor to apply to (exponential) moving average prediction
    """
    def __init__(self, name, n_units, alpha, **kwargs):
        super().__init__(name, **kwargs)

        ## RPE meta-parameters
        self.alpha = alpha

        ## Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.mu = Compartment(restVals) ## reward predictor state(s)
        self.reward = Compartment(restVals) ## target reward signal(s)
        self.rpe = Compartment(restVals) ## reward prediction error(s)

    @staticmethod
    def _advance_state(dt, alpha, mu, rpe, reward):
        ## compute/update RPE and predictor values
        rpe = reward - mu
        mu = mu * (1. - alpha) + reward * alpha
        return mu, rpe

    @resolver(_advance_state)
    def advance_state(self, mu, rpe):
        self.mu.set(mu)
        self.rpe.set(rpe)

    @staticmethod
    def _reset(batch_size, n_units):
        mu = jnp.zeros((batch_size, n_units)) #None
        rpe = jnp.zeros((batch_size, n_units)) #None
        return mu, rpe

    @resolver(_reset)
    def reset(self, mu, rpe):
        self.mu.set(mu)
        self.rpe.set(rpe)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "RewardErrorCell - computes the reward prediction error "
                         "at each time step `t`; this is an online RPE estimator"
        }
        compartment_props = {
            "input_compartments":
                {"reward": "External reward signals/values"},
            "output_compartments":
                {"mu": "Current state of reward predictor",
                 "rpe": "Current value of reward prediction error at time `t`",},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "alpha": "Moving average decay factor"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "rpe = reward - mu; mu = mu * (1 - alpha) + reward * alpha",
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
        X = GaussianErrorCell("X", 9)
    print(X)
