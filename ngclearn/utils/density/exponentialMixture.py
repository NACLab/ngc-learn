from jax import numpy as jnp, random, jit, scipy
from functools import partial
import time, sys
import numpy as np

from ngclearn.utils.density.mixture import Mixture

########################################################################################################################
## internal routines for mixture model
########################################################################################################################

@jit
def _log_exponential_pdf(X, rate):
    """
    Calculates the multivariate exponential log likelihood of a design matrix/dataset `X`, under a given parameter 
    probability `p`.

    Args:
        X: a design matrix (dataset) to compute the log likelihood of

        rate: a parameter rate vector

    Returns:
        the log likelihood (scalar) of this design matrix X
    """
    #D = X.shape[1] * 1. ## get dimensionality
    ## pdf(x; r) = r * np.exp(-r * x), where r is "rate"
    ## log (r exp(-r x) ) = log(r) + log(exp(-r x) = log(r) - r x
    vec_ll = -(X * rate) + jnp.log(rate) ## log exponential
    log_ll = jnp.sum(vec_ll, axis=1, keepdims=True) ## get per-datapoint LL
    return log_ll

@jit
def _calc_exponential_pdf_vals(X, p):
    log_ll = _log_exponential_pdf(X, p) ## get log-likelihood
    ll = jnp.exp(log_ll) ## likelihood
    return log_ll, ll

@jit
def _calc_priors_and_rates(X, weights, pi): ## M-step co-routine
    ## calc new rates, responsibilities, and priors given current stats
    N = X.shape[0]  ## get number of samples
    ## calc responsibilities
    r = (pi * weights)
    r = r / jnp.sum(r, axis=1, keepdims=True) ## responsibilities
    _pi = jnp.sum(r, axis=0, keepdims=True) / N ## calc new priors
    ## calc weighted rates (weighted by responsibilities)
    Z = jnp.sum(r, axis=0, keepdims=True) ## calc partition function
    M = (Z > 0.) * 1.
    Z = Z * M + (1. - M) ## we mask out any zero partition function values
    rates = jnp.matmul(r.T, X) / Z.T
    return rates, _pi, r

@partial(jit, static_argnums=[1])
def _sample_prior_weights(dkey, n_samples, pi): ## samples prior weighting parameters (of mixture)
    log_pi = jnp.log(pi)  ## calc log(prior)
    lats = random.categorical(dkey, logits=log_pi, shape=(n_samples, 1))  ## sample components/latents
    return lats

@partial(jit, static_argnums=[1])
def _sample_component(dkey, n_samples, rate): ## samples a component (of mixture)
    ## sampling ~[exp(rx)] is same as r * [~exp(x)]
    eps = jax.random.exponential(dkey, shape=(n_samples, mu.shape[1])) * rate ## draw exponential samples
    return x_s

########################################################################################################################

class ExponentialMixture(Mixture): ## Exponential mixture model (mixture-of-exponentials)
    """
    Implements a exponential mixture model (EMM) -- or mixture of exponentials (MoExp).
    Adaptation of parameters is conducted via the Expectation-Maximization (EM)
    learning algorithm. Note that this exponential mixture assumes that each component 
    is a factorizable mutlivariate exponential distribution. (A Categorical distribution 
    is assumed over the latent variables).

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
        self.key, *skey = random.split(self.key, 3)
        self.pi = jnp.ones((1, self.K)) / (self.K * 1.)
        ptrs = random.permutation(skey[0], X.shape[0])
        self.rate = [] 
        for j in range(self.K):
            ptr = ptrs[j]
            self.key, *skey = random.split(self.key, 3)
            eps = random.uniform(skey[0], minval=0., maxval=0.5, shape=(1, dim)) ## jitter initial rate params
            self.rate.append(eps)

    def calc_log_likelihood(self, X):
        """
        Calculates the multivariate exponential log likelihood of a design matrix/dataset `X`, under the current
        parameters of this exponential mixture.

        Args:
            X: the design matrix to estimate log likelihood values over under this EMM

        Returns:
            (column) vector of individual log likelihoods, scalar for the complete log likelihood p(X)
        """
        ll = 0.
        for j in range(self.K):
            log_ll_j, ll_j = _calc_exponential_pdf_vals(X, self.rate[j])
            ll = ll_j + ll
        log_ll = jnp.log(ll) ## vector of individual log p(x_n) values
        complete_ll = jnp.sum(log_ll) ## complete log-likelihood for design matrix X, i.e., log p(X)
        return log_ll, complete_ll

    def _E_step(self, X): ## Expectation (E) step, co-routine
        weights = []
        for j in range(self.K):
            log_ll_j, ll_j = _calc_exponential_pdf_vals(X, self.rate[j])
            weights.append( ll_j )
        weights = jnp.concat(weights, axis=1)
        return weights ## data-dependent weights (intermediate responsibilities)

    def _M_step(self, X, weights): ## Maximization (M) step, co-routine
        rates, pi, r = _calc_priors_and_rates(X, weights, self.pi)
        self.pi = pi ## store new prior parameters
        # calc weighted covariances
        for j in range(self.K):
            #r_j = r[:, j:j + 1]
            rate_j = rates[j:j + 1, :]
            self.rate[j] = rate_j ## store new rate(j) parameter
        return rates, r

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
            self.update(X) ## carry out one E-step followed by an M-step
            rates = jnp.concat(self.rate, axis=0)
            dor = jnp.linalg.norm(rates - rates_prev) ## norm of difference-of-rates
            if verbose:
                print(f"{i}: Rate-diff = {dor}")
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
        """
        r_w = self._E_step(X)  ## carry out E-step
        rates, respon = self._M_step(X, r_w) ## carry out M-step

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
        ## sample prior
        self.key, *skey = random.split(self.key, 3)
        if mode_j >= 0: ## sample from a particular mode / component
            rate_j = self.rate[mode_j]
            Xs = _sample_component(skey[0], n_samples=n_samples, rate=rate_j)
        else: ## sample from full mixture distribution
            ## sample components/latents
            lats = _sample_prior_weights(skey[0], n_samples=n_samples, pi=self.pi)
            ## then sample chosen component exponential
            Xs = []
            for j in range(self.K):
                freq_j = int(jnp.sum((lats == j)))  ## compute frequency over mode
                self.key, *skey = random.split(self.key, 3)
                x_s = _sample_component(skey[0], n_samples=freq_j, rate=self.rate[j])
                Xs.append(x_s)
            Xs = jnp.concat(Xs, axis=0)
        return Xs

