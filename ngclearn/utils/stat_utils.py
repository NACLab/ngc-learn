"""
Statistical functions/utilities file.
"""
import tensorflow as tf
import numpy as np
seed = 69
#tf.random.set_random_seed(seed=seed)
tf.random.set_seed(seed=seed)
np.random.seed(seed)


def sample_uniform(n_s, n_dim):
    """
    Samples a multivariate Uniform distribution

    Args:
        n_s: number of samples to draw

        n_dim: dimensionality of the sample space

    Returns:
        an (n_s x n_dim) matrix of uniform samples (one vector sample per row)
    """
    eps = tf.random.uniform(shape=(n_s,n_dim), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    return eps

def sample_gaussian(n_s, mu=0.0, sig=1.0, n_dim=-1):
    """
    Samples a multivariate Gaussian assuming a diagonal covariance or scalar
    variance (shared across dimensions) in the form of a standard deviation
    vector/scalar.

    Args:
        n_s: number of samples to draw

        mu: (1 x D) mean of the Gaussian distribution

        sig: (1 x D) or (1 x 1) standard deviation of the Gaussian distribution

        n_dim: dimensionality of the sample space

    Returns:
        an (n_s x n_dim) matrix of uniform samples (one vector sample per row)
    """
    dim = n_dim
    if dim <= 0:
        dim = mu.shape[1]
    eps = tf.random.normal([n_s, dim], mean=0.0, stddev=1.0, seed=seed)
    return mu + eps * sig

def sample_bernoulli(p):
    """
    Samples a multivariate Bernoulli distribution

    Args:
        p: probabilities to samples of shape (n_s x D)

    Returns:
        an (n_s x D) (binary) matrix of Bernoulli samples (one vector sample per row)
    """
    eps = tf.random.uniform(shape=p.shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    samples = tf.math.greater(p, eps)
    #samples = tf.math.less(p, eps)
    return tf.cast(samples,dtype=tf.float32)

def convert_to_spikes(x_data, gain=1.0, offset=0.0, n_trials=1):
    p = tf.clip_by_value(x_data, 0., 1.) * gain + offset
    samples = sample_bernoulli(p)
    for _ in range(n_trials-1):
        samples = sample_bernoulli(p)
    return samples

def calc_log_gauss_pdf(X, mu, cov):
    """
    Calculates the log Gaussian probability density function (PDF)

    Args:
        X: an (N x D) data design matrix to measure log density over

        mu: the (1 x D) vector mean of the Gaussian distribution

        cov: the (D x D) covariance matrix of the Gaussian distribution

    Returns:
        a (N x 1) column vector w/ each row containing log density value per sample
    """
    prec_chol = calc_prec_chol(mu, cov)
    n_samples, n_features = X.shape
    # det(precision_chol) is half of det(precision)
    log_det = tf.linalg.logdet(prec_chol) # log determinant of Cholesky precision
    y = tf.matmul(X, prec_chol) - tf.matmul(mu, prec_chol)
    log_prob = tf.reduce_sum(y * y, axis=1,keepdims=True)
    return -0.5 * (n_features * tf.math.log(2 * np.pi) + log_prob) + log_det

def calc_list_moments(data_list, num_dec=3):
    """
    Compute the mean and standard deviation from a list of data values. This is
    for simple scalar measurements/metrics that will be printed to I/O.

    Args:
        data_list: list of data values, each element should be (1 x 1)

        num_dec: number of decimal points to round values to (Default = 3)

    Returns:
        (mu, sigma), where mu = mean and sigma = standard deviation
    """
    mu = 0.0
    sigma = 0.0
    if len(data_list) > 0:
        for i in range(len(data_list)):
            log_px_i = data_list[i]
            mu += log_px_i
        mu = mu / len(data_list) * 1.0
    if len(data_list) > 1:
        for i in range(len(data_list)):
            log_px_i = data_list[i]
            sigma += (log_px_i - mu) * (log_px_i - mu)
        sigma = np.sqrt(sigma/ (len(data_list) * 1.0 - 1.0))
    mu = np.round(mu, num_dec)
    sigma = np.round(sigma, num_dec)
    return mu, sigma

def calc_covariance(X, mu_=None, weights=None, bias=True):
    """
    Calculate the covariance matrix of X

    Args:
        X: an (N x D) data design matrix to measure log density over
            (1 row vector - 1 data point)

        mu_: a pre-computed (1 x D) vector mean of the Gaussian distribution
            (Default = None)

        weights: a (N x 1) weighting column vector, one row is weight applied
            to one sample in X (Default = None)

        bias: (only applies if weights is None), if True, compute the
            biased estimator of covariance

    Returns:
        a (D x D) covariance matrix
    """
    eps = 1e-4 #1e-4
    Ie = tf.eye(X.shape[1]) * eps
    if weights is None:
        # calculate classical covariance
        mu = mu_
        if mu is None:
            mu = tf.reduce_mean(X,axis=0,keepdims=True)
        C = tf.subtract(X, mu)
        if bias is True:
            C = tf.matmul(tf.transpose(C), C)/(X.shape[0]*1.0) # computes correlation matrix / N (biased estimate)
        else:
            C = tf.matmul(tf.transpose(C), C)/(X.shape[0]*1.0 - 1.0) # computes correlation matrix / N-1
        C = C + Ie # Ie controls for singularities
    else:
        nk = tf.reduce_sum(weights) + 10.0 * np.finfo(resp.dtype).eps
        mu = mu_
        if mu is None:
            mu = (weights * X)/nk
        # calculcate weighted covariance
        diff = X - mu
        C = tf.matmul((weights * diff), diff,transpose_a=True) / nk
        C = C + Ie
    return C

def calc_gKL(mu_p, sigma_p, mu_q, sigma_q):
    """
    Calculate the Gaussian Kullback-Leibler (KL) divergence between two
    multivariate Gaussian distributions, i.e., KL(p||q).

    Args:
        mu_p: (1 x D) vector mean of distribution p

        sigma_p: (D x D) covariance matrix of distributon p

        mu_q: (1 x D) vector mean of distribution q

        sigma_q: (D x D) covariance matrix of distributon q

    Returns:
        the scalar KL divergence
    """
    # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    prec_q = ainv(sigma_q)# tf.linalg.pinv(sigma_q)
    k = mu_p.shape[1]
    term1 = (tf.linalg.logdet(sigma_q) - tf.linalg.logdet(sigma_p)) - k
    #term1 = tf.math.log(tf.linalg.det(sigma_q) - tf.linalg.det(sigma_p)) - k
    diff = mu_p - mu_q
    term2 = tf.matmul(tf.matmul(diff,prec_q),tf.transpose(diff))
    term3 = tf.linalg.trace(tf.matmul(prec_q, sigma_p))
    KL = (term1 + term2 + term3) * 0.5
    return KL

def ainv(A):
    """
    Computes the inverse of matrix A

    Args:
        A: matrix to invert

    Returns:
        the inversion of A
    """
    eps = 0.0001 # stability factor for precision/covariance computation
    cov_l = A
    diag_l = tf.eye(cov_l.shape[1])
    prec_l = tf.linalg.inv(cov_l + diag_l * eps)
    # # Note for Numerical Stability:
    # #   Add small pertturbation eps * I to covariance before decomposing
    # #   (due to rapidly decaying Eigen values)
    # #R = tf.linalg.cholesky(cov_l + diag_l) # decompose
    # R = tf.linalg.cholesky(cov_l + diag_l * eps) # decompose
    # #R = tf.linalg.cholesky(cov_l)
    # prec_l = tf.transpose(tf.linalg.triangular_solve(R,diag_l,lower=True))
    return prec_l
