import tensorflow as tf
import sys
import numpy as np

#from ngclearn.utils.stat_utils import calc_log_gauss_pdf
#from ngclearn.utils.transform_utils import softmax

seed = 69
tf.random.set_seed(seed=seed)

class NoisySinusoid:
    """
    Implements the noisy sinusoid stochastic (temporal) data generating process.
    Note that centered Gaussian noise is used to create the corrupted samples of
    the underlying sinusoidal process.

    Args:
        sigma: a (1 x D) vector (numpy) that dictates the standard deviation of the process

        dt: the integration time step (Default = 0.01)

        x_initial: the initial value of the process (Default = None, yielding a
            zero vector starting point)
    """
    def __init__(self, sigma, dt=0.01, x_initial=None):
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial # initial condition
        self.x_prev = None
        self.reset() # ensure at initialization this O-H process starts at the initial condition

    def sample(self):
        """
        Draws a sample from the current state of this noisy sinusoidal temporal process.

        Returns:
            a vector sample of shape (1 x D)
        """
        # increment the internal tracking of x
        x = self.x_prev + self.dt
        x = x.astype(np.float32)
        # compute the Gaussian/white-noise corrupted sine wave output
        y = np.sin(x) + np.random.normal(size=self.sigma.shape) * self.sigma
        y = y.astype(np.float32)
        # Store x sample into x_prev state
        self.x_prev = x # this step ensures that the next noise sample is dependent upon current one
        return y

    def reset(self):
        """
        Resets the temporal noise process back to its initial condition/starting point.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else: # reset the noise process back to zero
            self.x_prev = np.zeros_like(self.sigma)
        self.x_prev = self.x_prev.astype(np.float32)
