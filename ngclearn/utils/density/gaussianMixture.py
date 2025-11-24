from jax import numpy as jnp, random, jit, scipy
from functools import partial
import time, sys
import numpy as np

from ngclearn.utils.density.mixture import Mixture

########################################################################################################################
## internal routines for mixture model
########################################################################################################################

@partial(jit, static_argnums=[3])
def _log_gaussian_pdf(X, mu, Sigma, use_chol_prec=True):
    """
    Calculates the multivariate Gaussian log likelihood of a design matrix/dataset `X`, under a given parameter mean
    `mu` and parameter covariance `Sigma`.

    Args:
        X: a design matrix (dataset) to compute the log likelihood of
        mu: a parameter mean vector
        Sigma: a parameter covariance matrix
        use_chol_prec: should this routine use Cholesky-factor computation of the precision (Default: True)

    Returns:
        the log likelihood (scalar) of this design matrix X
    """
    D = mu.shape[1] * 1. ## get dimensionality
    if use_chol_prec: ## use Cholesky-factor calc of precision
        C = jnp.linalg.cholesky(Sigma) # calc_prec_chol(mu, cov)
        inv_C = jnp.linalg.pinv(C)
        precision = jnp.matmul(inv_C.T, inv_C)
    else: ## use Moore-Penrose pseudo-inverse calc of precision
        precision = jnp.linalg.pinv(Sigma)
    ## finish computing log-likelihood
    sign_ld, abs_ld = jnp.linalg.slogdet(Sigma)
    log_det_sigma = abs_ld * sign_ld ## log-determinant of precision
    Z = X - mu ## calc deltas
    quad_term = jnp.sum((jnp.matmul(Z, precision) * Z), axis=1, keepdims=True) ## LL quadratic term
    return -(jnp.log(2. * np.pi) * D + log_det_sigma + quad_term) * 0.5

@partial(jit, static_argnums=[3])
def _calc_gaussian_pdf_vals(X, mu, Sigma, use_chol_prec=True):
    log_likeli = _log_gaussian_pdf(X, mu, Sigma, use_chol_prec)
    likeli = jnp.exp(log_likeli)
    return log_likeli, likeli

@jit
def _calc_gaussian_mixture_stats(raw_likeli, pi):
    likeli = raw_likeli * pi
    gamma = likeli / jnp.sum(likeli, axis=1, keepdims=True)  ## responsibilities
    likeli = jnp.sum(likeli, axis=1, keepdims=True)  ## Sum_j[ pi_j * pdf_gauss(x_n; mu_j, Sigma_j) ]
    log_likeli = jnp.log(likeli)  ## vector of individual log p(x_n) values
    complete_log_likeli = jnp.sum(log_likeli)  ## complete log-likelihood for design matrix X, i.e., log p(X)
    return log_likeli, complete_log_likeli, gamma

@partial(jit, static_argnums=[3])
def _calc_weighted_cov(X, mu, weights, assume_diag_cov=False): ## M-step co-routine
    ## calc new covariance Sigma given data, means, and responsibilities
    diff = X - mu
    sigma_j = jnp.matmul((weights * diff).T, diff) / jnp.sum(weights)
    if assume_diag_cov:
        sigma_j = sigma_j * jnp.eye(sigma_j.shape[1])
    return sigma_j

@jit
def _calc_priors_and_means(X, weights, pi): ## M-step co-routine
    ## calc new means, responsibilities, and priors given current stats
    N = X.shape[0]  ## get number of samples
    ## calc responsibilities
    _pi = jnp.sum(weights, axis=0, keepdims=True) / N ## calc new priors
    ## calc weighted means (weighted by responsibilities)
    Z = jnp.sum(weights, axis=0, keepdims=True) ## partition function
    M = (Z > 0.) * 1.
    Z = Z * M + (1. + M) ## removes div-by-0 cases
    means = jnp.matmul(weights.T, X) / Z.T
    return _pi, means

@partial(jit, static_argnums=[1])
def _sample_prior_weights(dkey, n_samples, pi): ## samples prior weighting parameters (of mixture)
    log_pi = jnp.log(pi)  ## calc log(prior)
    lats = random.categorical(dkey, logits=log_pi, shape=(n_samples, 1))  ## sample components/latents
    return lats

@partial(jit, static_argnums=[1, 4])
def _sample_component(dkey, n_samples, mu, Sigma, assume_diag_cov=False): ## samples a component (of mixture)
    eps = random.normal(dkey, shape=(n_samples, mu.shape[1])) ## draw unit Gaussian noise
    ## apply scale-shift transformation
    if assume_diag_cov:
        R = jnp.sum(jnp.sqrt(Sigma), axis=0, keepdims=True)
        x_s = mu + eps * R
    else:
        R = jnp.linalg.cholesky(Sigma)  ## decompose covariance via Cholesky
        x_s = mu + jnp.matmul(eps, R)  # tf.matmul(eps, R)
    return x_s

# def _log_gaussian_pdf(X, mu, sigma):
#     C = jnp.linalg.cholesky(sigma) #calc_prec_chol(mu, cov)
#     inv_C = jnp.linalg.pinv(C)
#     prec_chol = jnp.matmul(inv_C, inv_C.T)
#     #prec_chol = jnp.linalg.inv(sigma)
#
#     N, D = X.shape ## n_samples x dimensionality
#     # det(precision_chol) is half of det(precision)
#     sign_ld, abs_ld = jnp.linalg.slogdet(prec_chol)
#     log_det = abs_ld * sign_ld ## log determinant of Cholesky precision
#     y = jnp.matmul(X, prec_chol) - jnp.matmul(mu, prec_chol)
#     log_prob = jnp.sum(y * y, axis=1, keepdims=True)
#     #return -0.5 * (D * jnp.log(np.pi * 2) + log_prob) + log_det
#     #return -0.5 * (D * jnp.log(np.pi * 2) + log_det + log_prob)
#     return -jnp.log(np.pi * 2) * (D * 0.5) - log_det * 0.5 - log_prob * 0.5

########################################################################################################################

class GaussianMixture(Mixture): ## Gaussian mixture model (mixture-of-Gaussians)
    """
    Implements a Gaussian mixture model (GMM) -- or mixture of Gaussians (MoG).
    Adaptation of parameters is conducted via the Expectation-Maximization (EM)
    learning algorithm and leverages full covariance matrices in the component
    multivariate Gaussians. (A Categorical distribution is assumed over the 
    latent variables).

    Args:
        K: the number of components/latent variables within this GMM

        max_iter: the maximum number of EM iterations to fit parameters to data (Default = 50)

        assume_diag_cov: if True, assumes a diagonal covariance for each component (Default = False)

        init_kmeans: <Unsupported>
    """
    # init_kmeans: if True, first learn use the K-Means algorithm to initialize
    #              the component Gaussians of this GMM (Default = False)

    def __init__(self, K, max_iter=50, assume_diag_cov=False, init_kmeans=False, key=None, **kwargs):
        super().__init__(K, max_iter, **kwargs) 
        self.K = K
        self.max_iter = int(max_iter)
        self.assume_diag_cov = assume_diag_cov
        self.init_kmeans = init_kmeans ## Unsupported currently
        self.mu = [] ## component mean parameters
        self.Sigma = [] ## component covariance parameters
        self.pi = None ## prior weight parameters
        #self.z_weights = None # variables for parameterizing weights for SGD
        self.key = random.PRNGKey(time.time_ns()) if key is None else key

    def init(self, X):
        """
        Initializes this GMM in accordance to a supplied design matrix.

        Args:
            X: the design matrix to initialize this GMM to

        """
        dim = X.shape[1]
        self.key, *skey = random.split(self.key, 3)
        self.pi = jnp.ones((1, self.K)) / (self.K * 1.)
        ptrs = random.permutation(skey[0], X.shape[0])
        for j in range(self.K):
            ptr = ptrs[j]
            #self.key, *skey = random.split(self.key, 3)
            self.mu.append(X[ptr:ptr+1,:])
            Sigma_j = jnp.eye(dim)
            #sigma_j = random.uniform(skey[0], minval=0.01, maxval=0.9, shape=(dim, dim))
            self.Sigma.append(Sigma_j)

    def calc_log_likelihood(self, X):
        """
        Calculates the multivariate Gaussian log likelihood of a design matrix/dataset `X`, under the current
        parameters of this Gaussian mixture model.

        Args:
            X: the design matrix to estimate log likelihood values over under this GMM

        Returns:
            (column) vector of individual log likelihoods, scalar for the complete log likelihood p(X)
        """
        likeli = []
        for j in range(self.K):
            _, likeli_j = _calc_gaussian_pdf_vals(X, self.mu[j], self.Sigma[j])
            likeli.append(likeli_j)
        likeli = jnp.concat(likeli, axis=1)
        log_likeli_vec, complete_log_likeli, gamma = _calc_gaussian_mixture_stats(likeli, self.pi)
        return log_likeli_vec, complete_log_likeli

    def _E_step(self, X): ## Expectation (E) step, co-routine
        likeli = []
        for j in range(self.K):
            _, likeli_j = _calc_gaussian_pdf_vals(X, self.mu[j], self.Sigma[j])
            likeli.append(likeli_j)
        likeli = jnp.concat(likeli, axis=1)
        log_likeli_vec, complete_log_likeli, gamma = _calc_gaussian_mixture_stats(likeli, self.pi)
        ## gamma => ## data-dependent weights (responsibilities)
        return gamma, log_likeli_vec, complete_log_likeli

    def _M_step(self, X, weights): ## Maximization (M) step, co-routine
        pi, means = _calc_priors_and_means(X, weights, self.pi)
        self.pi = pi ## store new prior parameters
        # calc weighted covariances
        for j in range(self.K):
            r_j = weights[:, j:j + 1] ## get j-th responsibility slice
            mu_j = means[j:j + 1, :]
            sigma_j = _calc_weighted_cov(X, mu_j, r_j, assume_diag_cov=self.assume_diag_cov)
            self.mu[j] = mu_j ## store new mean(j) parameter
            self.Sigma[j] = sigma_j ## store new covariance(j) parameter
        return pi, means

    def fit(self, X, tol=1e-3, verbose=False):
        """
        Run full fitting process of this GMM.

        Args:
            X: the dataset to fit this GMM to

            tol: the tolerance value for detecting convergence (via difference-of-means); will engage in early-stopping
                if tol >= 0. (Default: 1e-3)

            verbose: if True, this function will print out per-iteration measurements to I/O
        """
        means_prev = jnp.concat(self.mu, axis=0)
        for i in range(self.max_iter):
            gamma, pi, means, complete_loglikeli = self.update(X) ## carry out one E-step followed by an M-step
            #means = jnp.concat(self.mu, axis=0)
            dom = jnp.linalg.norm(means - means_prev) ## norm of difference-of-means
            if verbose:
                print(f"{i}: Mean-diff = {dom}  log(p(X)) = {complete_loglikeli} nats")
            #print(jnp.linalg.norm(means - means_prev))
            if tol >= 0. and dom < tol:
                print(f"Converged after {i + 1} iterations.")
                break
            means_prev = means

    def update(self, X):
        """
        Performs a single iterative update (E-step followed by M-step) of parameters (assuming model initialized)

        Args:
            X: the dataset / design matrix to fit this GMM to
        """
        gamma, _, complete_likeli = self._E_step(X)  ## carry out E-step
        pi, means = self._M_step(X, gamma) ## carry out M-step
        return gamma, pi, means, complete_likeli

    def sample(self, n_samples, mode_j=-1):
        """
        Draw samples from the current underlying GMM model

        Args:
            n_samples: the number of samples to draw from this GMM

            mode_j: if >= 0, will only draw samples from a specific component of this GMM
                (Default = -1), ignoring the Categorical prior over latent variables/components

        Returns:
            Design matrix of samples drawn under the distribution defined by this GMM
        """
        self.key, *skey = random.split(self.key, 3)
        if mode_j >= 0: ## sample from a particular mode
            mu_j = self.mu[mode_j]  ## directly select a specific component
            Sigma_j = self.Sigma[mode_j]
            Xs = _sample_component(
                skey[0], n_samples=n_samples, mu=mu_j, Sigma=Sigma_j, assume_diag_cov=self.assume_diag_cov
            )
        else: ## sample from full mixture distribution
            ## sample (prior) components/latents
            lats = _sample_prior_weights(skey[0], n_samples=n_samples, pi=self.pi)
            ## then sample chosen component Gaussian(s)
            Xs = []
            for j in range(self.K):
                freq_j = int(jnp.sum((lats == j)))  ## compute frequency over mode
                self.key, *skey = random.split(self.key, 3)
                x_s = _sample_component( ## now physically sample component
                    skey[0], n_samples=freq_j, mu=self.mu[j], Sigma=self.Sigma[j], assume_diag_cov=self.assume_diag_cov
                )
                Xs.append(x_s)
            Xs = jnp.concat(Xs, axis=0)
        return Xs

