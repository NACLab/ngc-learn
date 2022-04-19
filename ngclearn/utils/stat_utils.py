"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

"""
Statistical utilities function file

@author: Alexander Ororbia
"""
import tensorflow as tf
import numpy as np
seed = 69
#tf.random.set_random_seed(seed=seed)

tf.random.set_seed(seed=seed)
np.random.seed(seed)


def sample_uniform(n_s, n_dim):
    """ Samples a multivariate Uniform distribution """
    eps = tf.random.uniform(shape=(n_s,n_dim), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    return eps

def sample_gaussian(n_s, mu=0.0, sig=1.0, n_dim=-1):
    """
        Samples a multivariate Gaussian assuming at worst a diagonal covariance
    """
    dim = n_dim
    if dim <= 0:
        dim = mu.shape[1]
    eps = tf.random.normal([n_s, dim], mean=0.0, stddev=1.0, seed=seed)
    return mu + eps * sig

def calc_log_gauss_pdf(X, mu, cov):
    """ Calculates the log Gaussian PDF """
    prec_chol = calc_prec_chol(mu, cov)
    n_samples, n_features = X.shape
    # det(precision_chol) is half of det(precision)
    log_det = tf.linalg.logdet(prec_chol) # log determinant of Cholesky precision
    y = tf.matmul(X, prec_chol) - tf.matmul(mu, prec_chol)
    log_prob = tf.reduce_sum(y * y, axis=1,keepdims=True)
    return -0.5 * (n_features * tf.math.log(2 * np.pi) + log_prob) + log_det

def calc_list_moments(data_list, num_dec=3):
    """ Compute 1st and 2nd moments from a list of data values """
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
    """ Calculates the covariance matrix of X """
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
    """ Calculate the Gaussian Kullback-Leibler (KL) divergence """
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
    """Computes the inverse of matrix A"""
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
