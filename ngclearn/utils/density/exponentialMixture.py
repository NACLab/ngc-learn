from jax import numpy as jnp, random, jit, scipy
from functools import partial
import time, sys

from ngclearn.utils.density.mixture import Mixture

########################################################################################################################
## internal routines for mixture model
########################################################################################################################
@jit
def _log_exponential_pdf(X, lmbda):
    """
    Calculates the multivariate exponential log likelihood of a design matrix/dataset `X`, under a given parameter
    probability `p`.

    Args:
        X: a design matrix (dataset) to compute the log likelihood of

        lmbda: a parameter rate vector

    Returns:
        the log likelihood (scalar) of this design matrix X
    """
    log_pdf = -jnp.matmul(X, lmbda.T) + jnp.sum(jnp.log(lmbda.T), axis=0)
    return log_pdf

@jit
def _calc_exponential_mixture_stats(X, lmbda, pi):
    log_exp_pdf = _log_exponential_pdf(X, lmbda)
    log_likeli = log_exp_pdf + jnp.log(pi) ## raw log-likelihood
    likeli = jnp.exp(log_likeli) ## raw likelihood
    gamma = likeli / jnp.sum(likeli, axis=1, keepdims=True) ## responsibilities
    weighted_log_likeli = jnp.sum(log_likeli * gamma, axis=1, keepdims=True)  ## get weighted EMM log-likelihood
    complete_loglikeli = jnp.sum(weighted_log_likeli)  ## complete log-likelihood for design matrix X, i.e., log p(X)
    return log_likeli, likeli, gamma, weighted_log_likeli, complete_loglikeli

@jit
def _calc_priors_and_rates(X, weights, pi):  ## M-step co-routine
    ## compute updates to pi params
    Zk = jnp.sum(weights, axis=0, keepdims=True)  ## summed weights/responsibilities; 1 x K
    Z = jnp.sum(Zk)  ## partition function
    pi = Zk / Z
    ## compute updates to lmbda params
    Z = jnp.matmul(weights.T, X)
    lmbda = Zk.T / Z
    return pi, lmbda

@partial(jit, static_argnums=[1])
def _sample_prior_weights(dkey, n_samples, pi): ## samples prior weighting parameters (of mixture)
    log_pi = jnp.log(pi)  ## calc log(prior)
    lats = random.categorical(dkey, logits=log_pi, shape=(n_samples, 1))  ## sample components/latents
    return lats

@partial(jit, static_argnums=[1])
def _sample_component(dkey, n_samples, rate): ## samples a component (of mixture)
    ## sampling ~[exp(rx)] is same as r * [~exp(x)]
    x_s = random.exponential(dkey, shape=(n_samples, rate.shape[1])) * rate ## draw exponential samples
    return x_s

########################################################################################################################

class ExponentialMixture(Mixture): ## Exponential mixture model (mixture-of-exponentials)
    """
    Implements an exponential mixture model (EMM) -- or mixture of exponentials (MoExp). Adaptation of parameters is
    conducted via the Expectation-Maximization (EM) learning algorithm. Note that this exponential mixture assumes that
    each component is a factorizable mutlivariate exponential distribution. (A Categorical distribution is assumed over
    the latent variables).

    The exponential distribution of each component (dimension `d`) is assumed to be: 

    | pdf(x_d; lmbda_d) = lmbda_d * exp(-lmbda_d x_d) for x >= 0, else 0 for x < 0; 
    | where lbmda is the rate parameter vector

    Args:
        K: the number of components/latent variables within this EMM

        max_iter: the maximum number of EM iterations to fit parameters to data (Default = 50)

        init_kmeans: <Unsupported>
    """

    def __init__(self, K, max_iter=50, init_kmeans=False, key=None, **kwargs):
        super().__init__(K, max_iter, **kwargs)
        self.K = K
        self.max_iter = int(max_iter)
        self.init_kmeans = init_kmeans ## Unsupported currently
        self.rate = [] ## component rate parameters
        self.pi = None ## prior weight parameters
        #self.z_weights = None # variables for parameterizing weights for SGD
        self.key = random.PRNGKey(time.time_ns()) if key is None else key

    def init(self, X):
        """
        Initializes this EMM in accordance to a supplied design matrix.

        Args:
            X: the design matrix to initialize this EMM to

        """
        dim = X.shape[1]
        self.key, *skey = random.split(self.key, 4)
        ## Computed jittered initial phi param values
        #self.pi = jnp.ones((1, self.K)) / (self.K * 1.)
        pi = jnp.ones((1, self.K))
        eps = random.uniform(skey[0], minval=0.99, maxval=1.01, shape=(1, self.K))
        pi = pi * eps
        self.pi = pi / jnp.sum(pi)

        ## Computed jittered initial rate (lmbda) param values
        lmbda_h = 1.0/jnp.mean(X, axis=0, keepdims=True)
        lmbda = random.uniform(skey[1], minval=0.99, maxval=1.01, shape=(self.K, dim)) * lmbda_h
        self.rate = [] 
        for j in range(self.K): ## set rates/lmbdas
            self.rate.append(lmbda[j:j+1, :])

    def calc_log_likelihood(self, X):
        """
        Calculates the multivariate exponential log likelihood of a design matrix/dataset `X`, under the current
        parameters of this exponential mixture.

        Args:
            X: the design matrix to estimate log likelihood values over under this EMM

        Returns:
            (column) vector of individual log likelihoods, scalar for the complete log likelihood p(X)
        """
        pi = self.pi ## get prior weight values
        lmbda = jnp.concat(self.rate, axis=0) ## get rates as a block matrix
        ## compute relevant log-likelihoods/likelihoods
        log_ll, ll, gamma, weighted_loglikeli, complete_likeli = _calc_exponential_mixture_stats(X, lmbda, pi)
        return weighted_loglikeli, complete_likeli

    def _E_step(self, X): ## Expectation (E) step, co-routine
        pi = self.pi ## get prior weight values
        lmbda = jnp.concat(self.rate, axis=0) ## get rates as a block matrix
        _, _, gamma, weighted_loglikeli, complete_likeli = _calc_exponential_mixture_stats(X, lmbda, pi)
        ## Note: responsibility weights gamma have shape => N x K
        return gamma, weighted_loglikeli, complete_likeli

    def _M_step(self, X, weights): ## Maximization (M) step, co-routine
        ## compute updates to pi and lmbda params
        pi, lmbda = _calc_priors_and_rates(X, weights, self.pi)
        self.pi = pi  ## store new prior parameters
        for j in range(self.K): ## store new rate/lmbda parameters
            self.rate[j] = lmbda[j:j+1, :]
        return pi, lmbda

    def fit(self, X, tol=1e-3, verbose=False):
        """
        Run full fitting process of this EMM.

        Args:
            X: the dataset to fit this EMM to

            tol: the tolerance value for detecting convergence (via difference-of-means); will engage in early-stopping
                if tol >= 0. (Default: 1e-3)

            verbose: if True, this function will print out per-iteration measurements to I/O
        """
        rates_prev = jnp.concat(self.rate, axis=0)
        for i in range(self.max_iter):
            gamma, pi, rates, complete_loglikeli = self.update(X) ## carry out one E-step followed by an M-step
            #rates = jnp.concat(self.rate, axis=0)
            dor = jnp.linalg.norm(rates - rates_prev) ## norm of difference-of-rates
            if verbose:
                print(f"{i}: Rate-diff = {dor}  log(p(X)) = {complete_loglikeli} nats")
            #print(jnp.linalg.norm(rates - rates_prev))
            if tol >= 0. and dor < tol:
                print(f"Converged after {i + 1} iterations.")
                break
            rates_prev = rates

    def update(self, X):
        """
        Performs a single iterative update (E-step followed by M-step) of parameters (assuming model initialized)

        Args:
            X: the dataset / design matrix to fit this BMM to

        Returns:
            responsibilities (gamma), priors (pi), rates (lambda), EMM log-likelihood
        """
        gamma, _, complete_log_likeli  = self._E_step(X)  ## carry out E-step
        pi, rates = self._M_step(X, gamma) ## carry out M-step
        return gamma, pi, rates, complete_log_likeli

    def sample(self, n_samples, mode_j=-1):
        """
        Draw samples from the current underlying EMM model

        Args:
            n_samples: the number of samples to draw from this EMM

            mode_j: if >= 0, will only draw samples from a specific component of this EMM
                (Default = -1), ignoring the Categorical prior over latent variables/components

        Returns:
            Design matrix of samples drawn under the distribution defined by this EMM
        """
        self.key, *skey = random.split(self.key, 3)
        if mode_j >= 0: ## sample from a particular mode
            rate_j = self.rate[mode_j] ## directly select a specific component
            Xs = _sample_component(skey[0], n_samples=n_samples, rate=rate_j)
        else: ## sample from full mixture distribution
            ## sample (prior) components/latents
            lats = _sample_prior_weights(skey[0], n_samples=n_samples, pi=self.pi)
            ## then sample chosen component exponential(s)
            Xs = []
            for j in range(self.K):
                freq_j = int(jnp.sum((lats == j)))  ## compute frequency over mode
                self.key, *skey = random.split(self.key, 3)
                x_s = _sample_component(skey[0], n_samples=freq_j, rate=self.rate[j])
                Xs.append(x_s)
            Xs = jnp.concat(Xs, axis=0)
        return Xs

