import tensorflow as tf
import sys
import numpy as np

from ngclearn.utils.stat_utils import calc_log_gauss_pdf
from ngclearn.utils.transform_utils import softmax

seed = 69
tf.random.set_seed(seed=seed)

class MoG:
    """
    Implements a mixture of Gaussians (MoG) stochastic data generating process.

    Args:
        x_dim: the dimensionality of the simulated data/input space

        num_comp: the number of components/latent variables within this GMM

        assume_diag_cov: if True, assumes a diagonal covariance for each component
            (Default = False)

        means: a list of means, each a (1 x D) vector (in tensor tf.float32 format)

        covar: a list of covariances, each a (D x D) vector (in tensor tf.float32 format)

        assume_diag_cov: if covar is None, forces auto-created covariance matrices
            to be strictly diagonal

        fscale: if covar is None, this controls the global scale of each component's
            covariance (Default = 1.0)

        seed: integer seed to control determinism of the underlying data generating process
    """
    def __init__(self, x_dim=2, num_comp=1, means=None, covar=None, phi=None,
                 assume_diag_cov=False, fscale=1.0, seed=69):
        num_comp_ = num_comp
        x_dim_ = x_dim
        if means is not None:
            num_comp_ = len(means)
            x_dim_ = means[0].shape[1]
        elif covar is not None:
            num_comp_ = len(covar)
            x_dim_ = means[0].shape[1]
        self.seed = seed
        self.x_dim = x_dim_
        self.num_comp = num_comp_
        self.assume_diag_cov = assume_diag_cov
        self.phi = phi
        if self.phi is None:
            self.phi = tf.ones([1,self.num_comp]) * (1.0/(self.num_comp * 1.0))
        self.mu = []
        self.sigma = []
        # initialize parameters
        for k in range(self.num_comp):
            if means is not None:
                self.mu.append(means[k])
            else:
                self.mu.append( tf.random.normal([1, x_dim], mean=0.0, stddev=fscale, seed=self.seed) )
            if covar is not None:
                self.sigma.append(covar[k])
            else:
                I = tf.eye(x_dim) # identity matrix
                sigma_k = tf.random.normal([x_dim, x_dim], mean=0.0, stddev=fscale, seed=self.seed)
                sigma_k = tf.math.abs(sigma_k * I) + sigma_k * (1.0 - I)
                self.sigma.append( sigma_k )

    def sample(self, n_s, mode_idx=-1):
        """
        Draw samples from the current underlying data generating mixture process.

        Args:
            n_s: the number of samples to draw from this GMM

            mode_i: if >= 0, will only draw samples from a selected, specific
                component of this process (Default = -1)
        """
        labels = None
        samples = None
        if mode_idx < 0:
            # sample multinomial latents
            pi = tf.zeros([n_s, self.phi.shape[1]]) + self.phi
            lats = tf.random.categorical(pi, num_samples=1)
            freq = []
            for i in range(self.num_comp):
                comp_i = tf.cast(tf.equal(lats, i),dtype=tf.float32)
                freq_i = int(tf.reduce_sum(comp_i))
                freq.append(freq_i)
            # given chosen modes/latents, sample corresponding component Gaussians
            for i in range(self.num_comp):
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
                    cov_i = self.sigma[i]
                    R = tf.linalg.cholesky(cov_i) # decompose covariance via Cholesky
                    x_s = mu_i + tf.matmul(eps,R) #,transpose_b=True)
                if i > 0:
                    samples = tf.concat([samples, x_s],axis=0)
                    labels = tf.concat([labels,tf.ones([x_s.shape[0],1],dtype=tf.int32) * i],axis=0)
                else:
                    samples = x_s
                    labels = tf.ones([x_s.shape[0],1],dtype=tf.int32) * i
        else:
            mu_k = self.mu[mode_idx]
            sigma_k = self.sigma[mode_idx]
            eps = tf.random.normal([n_s, mu_k.shape[1]], mean=0.0, stddev=1.0, seed=self.seed)
            R = None
            if self.assume_diag_cov is True:
                R = tf.reduce_sum(tf.math.sqrt(sigma_k),axis=0,keepdims=True)
                x_s = mu_k + eps * R
            else:
                R = tf.linalg.cholesky(sigma_k) # decompose covariance via Cholesky
                x_s = mu_k + tf.matmul(eps,R)
            labels = tf.ones([x_s.shape[0],1],dtype=tf.int32) * mode_idx
            samples = x_s
        return samples, labels
