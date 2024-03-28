from jax import numpy as jnp, random, jit
from functools import partial
import time, sys
import numpy as np
#from sklearn import mixture
#from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
#from ngclearn.utils.stat_utils import calc_log_gauss_pdf
from ngclearn.utils.model_utils import softmax
#from kmeans import K_Means
from sklearn import mixture

#seed = 69
#tf.random.set_seed(seed=seed)

class GMM:
    """
    Implements a Gaussian mixture model (GMM) -- or mixture of Gaussians, MoG.
    Adaptation of parameters is conducted via the Expectation-Maximization (EM)
    learning algorithm and leverages full covariance matrices in the component
    multivariate Gaussians.

    Note this is a (JAX) wrapper model that houses the sklearn implementation for learning.
    The sampling process has been rewritten to utilize GPU matrix computation.

    Args:
        k: the number of components/latent variables within this GMM

        max_iter: the maximum number of EM iterations to fit parameters to data
            (Default = 5)

        assume_diag_cov: if True, assumes a diagonal covariance for each component
            (Default = False)

        init_kmeans: if True, first learn use the K-Means algorithm to initialize
            the component Gaussians of this GMM (Default = True)
    """
    def __init__(self, k, max_iter=5, assume_diag_cov=False, init_kmeans=True):
        self.use_sklearn = True
        self.k = k
        self.max_iter = int(max_iter)
        self.assume_diag_cov = assume_diag_cov
        self.init_kmeans = init_kmeans
        self.mu = []
        self.sigma = []
        self.prec = []
        self.weights = None
        self.z_weights = None # variables for parameterizing weights for SGD

    def fit(self, data):
        """
        Run full fitting process of this GMM.

        Args:
            data: the dataset to fit this GMM to
        """
        pass

    def update(self, X):
        """
        Performs a single iterative update of parameters (assuming model initialized)

        Args:
            X: the dataset / design matrix to fit this GMM to
        """
        pass

    def sample(self, n_s, mode_i=-1, samples_modes_evenly=False):
        """
        (Efficiently) Draw samples from the current underlying GMM model

        Args:
            n_s: the number of samples to draw from this GMM

            mode_i: if >= 0, will only draw samples from a specific component of this GMM
                (Default = -1), ignoring the Categorical prior over latent variables/components

            samples_modes_evenly: if True, will ignore the Categorical prior over latent
                variables/components and draw an approximately equal number of samples from
                each component
        """
        pass
