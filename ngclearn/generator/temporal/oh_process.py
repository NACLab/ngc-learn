import tensorflow as tf
import sys
import numpy as np

from ngclearn.utils.stat_utils import calc_log_gauss_pdf
from ngclearn.utils.transform_utils import softmax

seed = 69
tf.random.set_seed(seed=seed)

class OUNoise:
    """
    Implements an Ornstein Uhlenbeck (O-H) stochastic (temporal) data generating process.

    Args:
        mean: a (1 x D) vector (numpy) (mean of the process)

        std_deviation: a (1 x D) vector (numpy) (standard deviation of the process)

        theta: meta-parameter to control the drift term of the process (Default = 0.15)

        dt: the integration time step (Default = 1e-2)

        x_initial: the initial value of the process (Default = None, yielding a
            zero vector starting point)
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial # initial condition
        self.x_prev = None
        self.reset() # ensure at initialization this O-H process starts at the initial condition

    def sample(self):
        """
        Draws a sample from the current state of this O-H temporal process.

        Returns:
            a vector sample of shape (1 x D)
        """
        # Formula below is adapted from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        x = x.astype(np.float32)
        # Store x sample into x_prev state
        self.x_prev = x # this step ensures that the next noise sample is dependent upon current one
        return x

    def reset(self):
        """
        Resets the temporal noise process back to its initial condition/starting point.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else: # reset the noise process back to zero
            self.x_prev = np.zeros_like(self.mean)
        self.x_prev = self.x_prev.astype(np.float32)
