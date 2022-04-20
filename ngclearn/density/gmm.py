"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

import tensorflow as tf
import sys
import numpy as np
#from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'models/')
from ngclearn.utils.stat_utils import calc_log_gauss_pdf
from ngclearn.utils.transform_utils import softmax
#from kmeans import K_Means
from sklearn import mixture


seed = 69
tf.random.set_seed(seed=seed)

"""
    Implements a custom/pure-TF Gaussian mixture model -- or mixture of Gaussians, MoG.
    Adaptation of parameters is conducted via the Expectation-Maximization (EM)
    learning algorithm and leverages full covariance matrices in the component
    multivariate Gaussians.

    Note this is a TF wrapper model that houses the sklearn implementation for learning.
    The sampling process has been rewritten to utilize GPU matrix computation.

    @author Alexander Ororbia
"""
class GMM:
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
        print(" ===== Start internal EM-fitting process ===== ")
        gmm = mixture.GaussianMixture(n_components=self.k,init_params="kmeans",
                                      covariance_type="full",verbose=2,warm_start=True)
        data = data.numpy()
        gmm.fit(data)
        print("Phi =\n",np.round(gmm.weights_, 6) )
        self.init_from_ScikitLearn(gmm)
        print(" ============================================= ")

    def init_from_ScikitLearn(self, gmm):
        '''
            Creates a GMM from a pre-trained Scikit-Learn model -- conversion
            sets things up for a row-major form of sampling, i.e., s ~ mu_k + eps * (L_k^T)
            where k is the sampled component index
        '''
        eps = 0.0001
        self.k = gmm.weights_.shape[0]
        self.mu = []
        self.sigma = []
        self.prec = []
        self.R = [] # cholesky-decomp of covariances (pre-computation)

        self.weights = tf.expand_dims(tf.cast(gmm.weights_, dtype=tf.float32),axis=0)
        for ki in range(gmm.means_.shape[0]):
            mu_i = tf.expand_dims(tf.cast(gmm.means_[ki],dtype=tf.float32),axis=0)
            #print("mu_i.shape = ", mu_i.shape)
            prec_i = tf.cast(gmm.precisions_[ki],dtype=tf.float32)
            cov_i = tf.cast(gmm.covariances_[ki],dtype=tf.float32)
            diag_i = tf.eye(cov_i.shape[1])
            #print("prec_i.shape = ", prec_i.shape)
            self.mu.append(mu_i)
            self.sigma.append(cov_i)
            self.prec.append(prec_i)
            # Note for Numerical Stability: Add small pertturbation eps * I to covariance before decomposing (due to rapidly decaying Eigen values)
            self.R.append( tf.transpose(tf.linalg.cholesky(cov_i + diag_i * eps))  ) # L^T is stored b/c we sample from a row-major perspective



    def estimate_log_prob(self, X):
        log_px = 0.0
        log_weights = tf.math.log(self.weights)
        for i in range(self.k):
            log_pi_i = log_weights[:,i]
            mu_i = self.mu[i]
            cov_i = self.sigma[i]
            log_pdf_i = calc_log_gauss_pdf(X,mu_i,cov_i)
            log_px += (log_pdf_i - log_pi_i)
        return tf.reduce_mean(tf.math.reduce_logsumexp(log_px, axis=1, keepdims=True))


    def estimate_gaussian_parameters(self, X, resp):
        eps = 1e-4 #1e-4
        Ie = tf.eye(X.shape[1]) * eps
        # X - n_samples x n_feats
        # resp - n_samples x n_components
        nk = tf.reduce_sum(resp,axis=0,keepdims=True) + (0.00001) #10 * 1e-6 # np.finfo(resp.dtype).eps
        means = []
        covariances = []
        for i in range(self.k):
            # calc weighted
            resp_i = tf.expand_dims(resp[:,i],axis=1)
            nk_i = tf.reduce_sum(resp_i)

            mu_i = tf.reduce_sum(resp_i * X,axis=0,keepdims=True)/nk_i
            # calc weighted covariance
            diff = X - mu_i
            C = tf.matmul((resp_i * diff), diff,transpose_a=True) / nk_i
            C = C + Ie
            cov_i = C
            means.append(mu_i)
            covariances.append(cov_i)

        return nk, means, covariances

    def m_step(self, X, log_resp):
        n_samples, _ = X.shape
        weights, mu, sigma = self.estimate_gaussian_parameters(X, tf.math.exp(log_resp))
        self.mu = mu
        self.sigma = sigma
        self.weights = weights / (n_samples * 1.0)

    def e_step(self, X):
        # calculate weighted log probabilities
        w_log_probs = self.calc_w_log_prob(X)
        # calculate weighted log responsibily normalization constant, Z
        log_prob_norm = tf.math.reduce_logsumexp(w_log_probs, axis=1, keepdims=True)
        log_resp = w_log_probs - log_prob_norm
        # return mean of log prob (normalized values) as a measure of the lower bound
        log_prob_norm = tf.reduce_mean(log_prob_norm)
        return log_prob_norm, log_resp, w_log_probs

    def update(self, X):
        """Performs a single iterative update of parameters (assuming model initialized)"""
        log_prob_norm, log_resp = self.e_step(X)
        self.m_step(X, log_resp)

    def calc_prob(self, X):
        """
            Computes probabilities p(z|x) of data samples in X under this GMM
        """
        w_log_probs = self.calc_w_log_prob(X)
        log_prob_norm = tf.math.reduce_logsumexp(w_log_probs, axis=1, keepdims=True)
        log_resp = w_log_probs - log_prob_norm
        probs = tf.math.exp(log_resp) # exponentiate the logit
        return probs

    def calc_gaussian_logpdf(self, X):
        """
            Calculates log densities/probabilities of data X under each component given this GMM
        """
        log_probs = None
        for i in range(self.k):
            mu_i = self.mu[i]
            sigma_i = self.sigma[i]
            ll_i = calc_log_gauss_pdf(X, mu_i, sigma_i)
            # calc log prob for i
            if i > 0:
                log_probs = tf.concat([log_probs,ll_i],axis=1)
            else:
                log_probs = ll_i
        return log_probs

    def calc_w_log_prob(self, X):
        """
            Calculates weighted log probabilities of data X under each component given this GMM
        """
        log_weights = tf.math.log(self.weights)
        w_log_probs = None
        for i in range(self.k):
            mu_i = self.mu[i]
            sigma_i = self.sigma[i]
            ll_i = calc_log_gauss_pdf(X, mu_i, sigma_i)
            # calc weight log prob for i
            log_w_i = log_weights[:,i]
            log_w_p_i = ll_i + log_w_i # likelihood + log(weights)
            if i > 0:
                w_log_probs = tf.concat([w_log_probs,log_w_p_i],axis=1)
            else:
                w_log_probs = log_w_p_i
        return w_log_probs

    def predict(self, X):
        """
            Chooses which component samples in X are likely to belong to given p(z|x)
        """
        w_log_probs = self.calc_w_log_prob(X)
        #log_prob_norm = tf.math.reduce_logsumexp(w_log_probs, axis=1, keepdims=True)
        #log_resp = w_log_probs - log_prob_norm
        #pred = tf.argmax(log_resp, axis=1)
        pred = tf.argmax(w_log_probs, axis=1)
        return pred

    def sample(self, n_s, mode_i=-1, samples_modes_evenly=False):
        """(Efficiently) Draw samples from the current underlying GMM model"""
        samples = None
        labels = None
        if mode_i >= 0: # sample from a particular mode / component
            i = mode_i
            freq_i = n_s
            #print(" component {0} -> {1} samps".format(i, freq_i))
            mu_i = self.mu[i]
            # draw freq_i samples from Gaussian component i via reparameterization trick (Gaussian is a scale-shift distribution)
            eps = tf.random.normal([freq_i, mu_i.shape[1]], mean=0.0, stddev=1.0, seed=seed)
            if self.assume_diag_cov is True:
                cov_i = self.sigma[i]
                R = tf.reduce_sum(tf.math.sqrt(cov_i),axis=0,keepdims=True)
                x_s = mu_i + eps * R
            else:
                if len(self.prec) == 0:
                    cov_i = self.sigma[i]
                    R = tf.linalg.cholesky(cov_i) # decompose covariance via Cholesky
                else:
                    R = self.R[i]
                x_s = mu_i + tf.matmul(eps,R) #,transpose_b=True)
            samples = x_s
            labels = tf.ones([x_s.shape[0],1],dtype=tf.int32) * i
        else: # sample from the full MoG (mixture of Gaussians)
            if samples_modes_evenly is True:
                print(" => Even sampler...")
                n_total_samp = n_s
                n_samp_per_mode = int(n_total_samp/self.weights.shape[1])
                leftover = n_total_samp - (n_samp_per_mode * self.weights.shape[1])
                print("  N_samp per mode = ",n_samp_per_mode)
                print("  Leftover = ",leftover)
                freq = [n_samp_per_mode] * self.weights.shape[1]
                freq[len(freq)-1] += leftover
            else:
                phi = self.weights
                pi = tf.zeros([n_s, phi.shape[1]]) + phi

                # sample multinomial latents
                #mode = tf.squeeze(tf.random.categorical(pi, num_samples=1)).numpy()
                lats = tf.random.categorical(pi, num_samples=1)
                freq = []
                for i in range(self.k):
                    comp_i = tf.cast(tf.equal(lats, i),dtype=tf.float32)
                    freq_i = int(tf.reduce_sum(comp_i))
                    freq.append(freq_i)

            # given chosen modes/latents, sample corresponding component Gaussians
            for i in range(self.k):
                freq_i = freq[i]
                #print(" component {0} -> {1} samps".format(i, freq_i))
                mu_i = self.mu[i]
                # draw freq_i samples from Gaussian component i via reparameterization trick (Gaussian is a scale-shift distribution)
                eps = tf.random.normal([freq_i, mu_i.shape[1]], mean=0.0, stddev=1.0, seed=seed)
                if self.assume_diag_cov is True:
                    cov_i = self.sigma[i]
                    R = tf.reduce_sum(tf.math.sqrt(cov_i),axis=0,keepdims=True)
                    x_s = mu_i + eps * R
                else:
                    if len(self.prec) == 0:
                        cov_i = self.sigma[i]
                        R = tf.linalg.cholesky(cov_i) # decompose covariance via Cholesky
                    else:
                        R = self.R[i]
                    x_s = mu_i + tf.matmul(eps,R) #,transpose_b=True)
                if i > 0:
                    samples = tf.concat([samples, x_s],axis=0)
                    labels = tf.concat([labels,tf.ones([x_s.shape[0],1],dtype=tf.int32) * i],axis=0)
                else:
                    samples = x_s
                    labels = tf.ones([x_s.shape[0],1],dtype=tf.int32) * i

        return samples, labels
