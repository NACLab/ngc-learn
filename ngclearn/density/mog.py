import tensorflow as tf
import sys
import numpy as np
#from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'models/')
from utils import calc_covariance, gaussian_ll, gaussian_pdf, calc_log_gauss_pdf, softmax
from kmeans import K_Means

seed = 69
tf.random.set_seed(seed=seed)

"""
    Implements a custom/pure-TF Gaussian mixture model -- or mixture of Gaussians, MoG.
    Adaptation of parameters is conducted via the Expectation-Maximization (EM)
    learning algorithm and leverages full covariance matrices in the component
    multivariate Gaussians.

    @author Alex Ororbia
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

    def init_from_ScikitLearn(self, gmm):
        '''
            Creates a GMM from a pre-trained Scikit-Learn model -- conversion
            set things up for a row-major form of sampling, i.e., s ~ mu_k + eps * (L_k^T)
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

    def compute_update(self, x):
        '''
            Computes the gradient of L(x) w.r.t. k-th component of GMM parameters
            (Based on page 45 of https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

            Some other resources:
            https://math.stackexchange.com/questions/1690926/differentiate-wrt-cholesky-decomposition
            https://math.stackexchange.com/questions/1942211/does-negative-transpose-sign-mean-inverse-of-a-transposed-matrix-or-transpose-of
        '''
        pi_prior = softmax(self.z_weights) # prior probs / mixing coefficients

        log_weights = tf.math.log(pi_prior) # get log prior probs (of latents)
        w_log_probs = None
        Z = 0.0 # normalization constant
        pi_post = None # posterior probs (of latents)
        for i in range(self.k):
            mu_i = self.mu[i]
            sigma_i = self.sigma[i]
            ll_i = calc_log_gauss_pdf(X, mu_i, sigma_i)
            # calc weight log prob for i
            log_w_i = log_weights[:,i]
            log_w_p_i = ll_i + log_w_i # likelihood + log(weights)
            pi_j = tf.math.exp(log_w_p_i)
            if i > 0:
                pi_post = tf.concat([pi_post, pi_j], axis=1)
            else:
                pi_post = pi_j
            Z += pi_j
        pi_post = pi_post / Z # normalize to get posterior probs

        # compute gradient w.r.t. PI (all k components at once)
        d_pi = pi_prior - pi_post
        pi_array = pi_post.numpy()

        d_mu = []
        d_Sigma = []
        for j in range(self.k):
            mu_j = self.mu[j]
            #Sigma_j = self.sigma[j]
            prec_j = self.prec[j]
            diff_j = x - mu_j
            p_j = pi_array[0,j]

            # compute gradient w.r.t. pi_j
            #pi_j = pi[j]
            #d_pi_j = p_j * (1.0 / pi_j)
            #d_pi.append(d_pi_j)

            # compute gradient w.r.t. mu_j
            d_mu_j = tf.matmul(diff_j, prec_j) * p_j
            d_mu.append( d_mu_j )

            # compute gradient w.r.t. Sigma_j (with Cholesky precision parameterization)
            B = tf.matmul(diff_j, diff_j,transpose_a=True)
            d_Sigma_j = tf.matmul(tf.matmul(prec_j, B), Prec_l) * (p_j * 0.5) - prec_j
            d_Sigma.append( d_Sigma_j )
            '''
            # compute gradient w.r.t. Sigma_j
            d_sigma_j = tf.matmul(tf.matmul(tf.matmul(sigma_j, diff_j),diff_j),sigma_j) * (w_term_j * 0.5) - sigma_j
            '''
        return d_pi, d_mu, d_Sigma


    def partial_fit(self, x, step_size=0.05):
        '''
            Computes an approximate gradient ascent update for GMM component parameters
        '''
        # compute approximate gradients for the current GMM
        d_z_weights, d_mu, d_Sigma = self.compute_update(x)
        self.weights = self.weights + d_z_weights * step_size
        for j in range(self.k):
            self.mu[j] = self.mu[j] - d_mu[j] * step_size
            cov_j = self.sigma[j] - d_Sigma[j] * step_size
            # apply constraints to covariance j
            diag_j = tf.eye(cov_j.shape[1])
            vari_j = tf.math.abs(cov_j * diag_j) # variance is restricted to be positive
            cov_j = vari_j + (cov_j * (1.0 - diag_j))
            self.sigma[j] = cov_j
            # compute precision matrix j
            R = tf.linalg.cholesky(cov_j) # Cholesky decomposition
            prec_j = tf.transpose(tf.linalg.triangular_solve(R,diag_j,lower=True))
            self.prec[j] = prec_j

    def fit_sgd(self, X, step_size=0.05, batch_size=32,n_iter=50):
        '''
            Updates GMM with Ororbia-style approximate gradient ascent
        '''
        X_numpy = X.numpy()
        lower_bound_tm1 = -10000.0
        for i in range(n_iter):
            ptrs = np.random.permutation(len(X_numpy))
            idx = 0
            n_seen = 0
            while n_seen < len(X_numpy):
                # compose mini-batch
                e_idx = idx + batch_size
                if e_idx > len(ptrs):
                    e_idx = len(ptrs)
                indices = ptrs[idx:e_idx]
                x_mb = tf.cast(X_numpy[indices],dtype=tf.float32)
                # update paramters given mini-batch of samples
                self.partial_fit(x_mb, step_size)
                n_seen += x_mb.shape[0]
            # Track model lower bound / progress
            lower_bound, _, _ = self.e_step(X)
            print(" L = {0}  in {1} EM steps...".format(lower_bound, iteration))
            if iteration > 0:
                delta = lower_bound - lower_bound_tm1
                if abs(delta) < tol_eps:
                    break
            lower_bound_tm1 = lower_bound
        print()

    def calc_centroid_stats(zLat, y_ind, n_C):
        clusters = cluster_by_class(zLat, y_ind, n_C)
        mu_by_cls = []
        cov_by_cls = []
        for i in range(len(clusters)):
            c_i = clusters[i]
            mu_i = tf.reduce_mean(c_i, axis=0, keepdims=True)
            mu_by_cls.append(mu_i)
            cov_i = calc_covariance(c_i, mu_i)
            cov_by_cls.append(cov_i)
        return clusters, mu_by_cls, cov_by_cls

    def initialize_(self, X, mu, sigma):
        self.shape = X.shape
        self.n, self.m = self.shape # n is num samples, m is num dimensions
        self.mu = mu
        self.sigma = sigma
        resp = tf.random.uniform([self.n,self.k], minval=0.0, maxval=1.0, seed=seed)
        resp = resp/tf.reduce_sum(resp,axis=1,keepdims=True)
        weights = tf.reduce_sum(resp,axis=0,keepdims=True) + 10 * 1e-6
        self.weights = weights/(self.n * 1.0)

    def initialize(self, X_):
        X = tf.cast(X_,dtype=tf.float32)
        self.shape = X.shape
        self.n, self.m = self.shape # n is num samples, m is num dimensions

        #self.phi = tf.ones([1,self.k]) * 1.0/(self.k * 1.0)
        #self.weights = tf.ones([self.n,self.k]) * 1.0/(self.k * 1.0)

        self.mu = []
        self.sigma = []
        self.prec = []

        if self.init_kmeans is True:
            if self.use_sklearn:
                print(" >> Running Scikit-Learn K-Means to init GMM model...")
                resp = np.zeros((self.n, self.k))
                label = KMeans(n_clusters=self.k, n_init=1, random_state=None).fit(X).labels_
                resp[np.arange(self.n), label] = 1
                resp = tf.cast(resp,dtype=tf.float32)
            else:
                print(" >> Running Classical K-Means to init GMM model...")
                resp = np.zeros((self.n, self.k))
                clf = K_Means(k=self.k)
                clf.fit(X)
                label = clf.predict(X)
                resp[np.arange(self.n), label] = 1
                resp = tf.cast(resp,dtype=tf.float32)
        else:
            #resp = tf.ones([self.n,self.k]) * 1.0/(self.k * 1.0)
            resp = tf.random.uniform([self.n,self.k], minval=0.0, maxval=1.0, seed=seed)
            resp = resp/tf.reduce_sum(resp,axis=1,keepdims=True)

        weights, means, covariances = self.estimate_gaussian_parameters(X, resp)
        self.weights = weights/(self.n * 1.0)
        self.mu = means
        self.sigma = covariances

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

    def fit(self, X):
        """
            Adapts parameters of GMM via Expectation-Maximization (EM)
        """
        tol_eps = 1e-6
        self.initialize(X)
        #L = self.calc_likelihood(X)
        #print(" L = {0}  in {1} EM steps...".format(L, -1))
        lower_bound_tm1 = -10000.0
        lower_bound = lower_bound_tm1
        iteration = -1
        for iteration in range(self.max_iter):
            # perform E-step
            log_prob_norm, log_resp, w_log_probs = self.e_step(X)
            # perform M-step
            self.m_step(X, log_resp)
            #print(self.weights)
            lower_bound = log_prob_norm #self.compute_lower_bound(log_resp, log_prob_norm)
            #lower_bound = self.estimate_log_prob(X)
            print(" L = {0}  in {1} EM steps...".format(lower_bound, iteration))
            if iteration > 0:
                delta = lower_bound - lower_bound_tm1
                if abs(delta) < tol_eps:
                    break
            lower_bound_tm1 = lower_bound
            #print(" -> {0} EM steps...".format(n_iter))
        lower_bound, _, _ = self.e_step(X) # final E-step to clean up model
        print(" L = {0}  in {1} EM steps...".format(lower_bound, iteration))
        print()
        #tf.print(tf.math.exp(self.calc_prob(X)),summarize=4)

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

    def sample(self, n_s):
        """(Efficiently) Draw samples from the current underlying GMM model"""
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

        samples = None
        labels = None
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
