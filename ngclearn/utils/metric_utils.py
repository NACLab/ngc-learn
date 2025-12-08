"""
Metric and measurement routines and co-routines. These functions are useful for model-level/simulation analysis as well
as experimental inspection and probing (many of these are neuroscience-oriented measurement functions).
"""
from jax import numpy as jnp, jit
from functools import partial
from sklearn.metrics import confusion_matrix, precision_score, recall_score

@partial(jit, static_argnums=[1])
def measure_fanoFactor(spikes, preserve_batch=False):
    """
    Calculates the Fano factor, i.e., a secondary statistics that probes the
    variability of a spike train within a particular time interval.

    Args:
        spikes: full spike train matrix; shape is (T x D) where D is number of
            neurons in a group/cluster

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar average score

    Returns:
        a 1 x D Fano factor vector (one factor per neuron) OR a single
        average Fano factor across the neuronal group
    """
    mu = jnp.mean(spikes, axis=0, keepdims=True)
    sigSqr = jnp.square(jnp.std(spikes, axis=0, keepdims=True))
    fano = sigSqr/mu
    if not preserve_batch:
        fano = jnp.mean(fano)
    return fano

@partial(jit, static_argnums=[1])
def measure_firingRate(spikes, preserve_batch=False):
    """
    Calculates the firing rate(s) of a group of neurons given full spike train.(s)

    Args:
        spikes: full spike train matrix; shape is (T x D) where D is number of
            neurons in a group/cluster

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar average score

    Returns:
        a 1 x D firing rate vector (one firing rate per neuron) OR a single
        average firing rate across the neuronal group
    """
    counts = jnp.sum(spikes, axis=0, keepdims=True)
    T = spikes.shape[0] * 1.
    fireRates = counts/T
    if not preserve_batch:
        fireRates = jnp.mean(fireRates)
    return fireRates

@partial(jit, static_argnums=[1])
def measure_breadth_TC(spikes, preserve_batch=False):
    """
    Calculates the breath tuning curve (BTC) of a group of neurons given full
    spike train.(s). BTC measures the neural selectivity such that the
    sparse code distribution concentrates near zero with a heavy tail. For a
    neural layer where most of the neurons fire, the activity distribution is
    more uniformly spread and BTC > 0.5. When most of the neurons do not fire,
    the firing distribution is peaked at zero and BTC < 0.5.

    Args:
        spikes: full spike train matrix; shape is (T x D) where D is number of
            neurons in a group/cluster

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar average score

    Returns:
        a 1 x D Fano factor vector (one factor per neuron) OR a single
        average Fano factor across the neuronal group
    """
    mu = jnp.mean(spikes, axis=0, keepdims=True)
    sigSqr = jnp.square(jnp.std(spikes, axis=0, keepdims=True))
    C = sigSqr/mu
    BTC = 1./(1 + jnp.square(C))
    if not preserve_batch:
        BTC = jnp.mean(BTC)
    return BTC

@jit
def measure_sparsity(codes, tolerance=0.):
    """
    Calculates the sparsity (ratio) of an input matrix, assuming each row within
    it is a non-negative vector.

    Args:
        codes: matrix (shape: N x D) of non-negative codes to measure
            sparsity of (per row)

        tolerance: lowest number to consider as "empty"/non-existent (Default: 0.)

    Returns:
        sparsity measurements per code (output shape: N x 1)
    """
    m = (codes > tolerance).astype(jnp.float32)
    rho = jnp.sum(m, axis=1, keepdims=True)/(codes.shape[1] * 1.)
    return rho

#@partial(jit, static_argnums=[2])
def analyze_scores(mu, y, extract_label_indx=True): ## examines classifcation statistics
    """
    Analyzes a set of prediction matrix and target/ground-truth matrix or vector.

    Args:
        mu: prediction (design) matrix; shape is (N x C) where C is number of classes
            and N is the number of patterns examined

        y: target / ground-truth (design) matrix; shape is (N x C) OR an array
            of class integers of length N (with "extract_label_indx = True")

        extract_label_indx: run an argmax to pull class integer indices from
            "y", assuming y is a one-hot binary encoding matrix (Default: True),
            otherwise, this assumes "y" is an array of class integer indices
            of length N

    Returns:
        confusion matrix, precision, recall, misses (empty predictions/all-zero rows),
        accuracy, adjusted-accuracy (counts all misses as incorrect)
    """
    miss_mask = (jnp.sum(mu, axis=1) == 0.) * 1.
    misses = jnp.sum(miss_mask) ## how many misses?
    labels = y
    if extract_label_indx:
        labels = jnp.argmax(y, axis=1)
    guesses = jnp.argmax(mu, axis=1)
    conf_matrix = confusion_matrix(labels, guesses)
    precision = precision_score(labels, guesses, average='macro')
    recall = recall_score(labels, guesses, average='macro')
    ## produce accuracy score measurements
    guess = jnp.argmax(mu, axis=1) ## gather all model/output guesses
    equality_mask = jnp.equal(guess, labels) * 1.
    ### compute raw accuracy
    acc = jnp.sum(equality_mask) / (y.shape[0] * 1.)
    ### compute hit-masked accuracy (adjusted accuracy
    adj_acc = jnp.sum(equality_mask * (1. - miss_mask)) / (y.shape[0] * 1.)
    ## output analysis statistics
    return conf_matrix, precision, recall, misses, acc, adj_acc

@partial(jit, static_argnums=[2])
def measure_ACC(mu, y, extract_label_indx=True): ## measures/calculates accuracy
    """
    Calculates the accuracy (ACC) given a matrix of predictions and matrix of targets.

    Args:
        mu: prediction (design) matrix; shape is (N x C) where C is number of classes
            and N is the number of patterns examined

        y: target / ground-truth (design) matrix; shape is (N x C) OR an array
            of class integers of length N (with "extract_label_indx = True")

        extract_label_indx: run an argmax to pull class integer indices from
            "y", assuming y is a one-hot binary encoding matrix (Default: True),
            otherwise, this assumes "y" is an array of class integer indices
            of length N

    Returns:
        scalar accuracy score
    """
    guess = jnp.argmax(mu, axis=1)
    if extract_label_indx:
        lab = jnp.argmax(y, axis=1)
    acc = jnp.sum( jnp.equal(guess, lab) )/(y.shape[0] * 1.)
    return acc

@partial(jit, static_argnums=[3])
def measure_BIC(X, n_model_params, max_model_score, is_log=True):
    """
    Measures the Bayesian information criterion (BIC) with respect to the final 
    score obtained by the model on a given dataset.

    | BIC = -2 ln(L) + K * ln(N); 
    | where N is number of data-points/rows of design matrix X, 
    | K is total number parameters of the model of interest, and 
    | L is the max/best-found value of a likelihood-like score L of the model

    Args:
        X: dataset/design matrix that a model was fit to (max-likelihood optimized) 

        n_model_params: total number of model parameters (int)

        max_model_score: max likelihood-like score obtained by model on X 

        is_log: is supplied `max_model_score` a log-likelihood? if this is False, 
            this metric will apply a natural logarithm of the score (Default: True)

    Returns: 
        scalar for the Bayesian information criterion score
    """
    ## BIC = K * ln(N) - 2 ln(L)
    L_hat = max_model_score ## model's likelihood-like score (at max point)
    K = n_model_params ## number of model params
    N = X.shape[0] ## number of data-points
    if not is_log:
        L_hat = jnp.log(L_hat) ## get log likelihood
    bic = -L_hat * 2. + jnp.log(N * 1.) * K
    return bic

@partial(jit, static_argnums=[2])
def measure_KLD(p_xHat, p_x, preserve_batch=False):
    """
    Measures the (raw) Kullback-Leibler divergence (KLD), assuming that the two input arguments contain valid
    probability distributions (in each row, if they are matrices). Note: If batch is preserved, this returns a column
    vector where each row is the KLD(x_pred, x_true) for that row's datapoint. (Further note that this function
    does not assume any particular distribution when calculating KLD)

    | Formula:
    | KLD(p_xHat, p_x) = (1/N) [ sum_i(p_x * jnp.log(p_x)) - sum_i(p_x * jnp.log(p_xHat)) ]
    | where sum_i implies summing across dimensions of vector-space of p_x

    Args:
        p_xHat: predicted probabilities; (N x C matrix, where C is number of categories)

        p_x: ground true probabilities; (N x C matrix, where C is number of categories)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    ## numerical control step
    offset = 1e-6
    _p_x = jnp.clip(p_x, offset, 1. - offset)
    _p_xHat = jnp.clip(p_xHat, offset, 1. - offset)
    ## calc raw KLD scores
    N = p_x.shape[1]
    term1 = jnp.sum(_p_x * jnp.log(_p_x), axis=1, keepdims=True) # * (1/N)
    term2 = -jnp.sum(_p_x * jnp.log(_p_xHat), axis=1, keepdims=True) # * (1/N)
    kld = (term1 + term2) * (1/N)  ## KLD-per-datapoint
    if not preserve_batch:
        kld = jnp.mean(kld)
    return kld

def measure_gaussian_KLD(mu1, Sigma1, mu2, Sigma2, use_chol_prec=True):
    """
    Calculates the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions, i.e.,
    KL(N(mu1, Sigma1) || N(mu2, Sigma2)).
    Formally, this means this routine calculates:

    | KL(N1 || N2) = [log(det(Sigma2)/det(Sigma1)) + trace(Prec2 * Sigma1) + (z * Prec2 * z) - D] * (1/2)
    | where N1 is the 1st Gaussian, i.e., N(mu1,Sigma1), and N2 is the 2nd Gaussian, i.e., N(mu2,Sigma2);
    | and where: Prec2 = (Sigma2)^{-1}, z = mu2 - mu1, and D is the data dimensionality

    Args:
        mu1: mean vector of first Gaussian distribution

        Sigma1: covariance matrix of first Gaussian distribution

        mu2: mean vector of second Gaussian distribution

        Sigma2: covariance matrix of second Gaussian distribution

        use_chol_prec: should this routine use Cholesky-factor computation of the precision of Sigma2 (Default: True)

    Returns:
        scalar representing KL-divergence between N(mu1, Sigma1) and N(mu2, Sigma2)
    """
    D = mu1.shape[1] ## dimensionality of data
    ## log(|Sigma2|/|Sigma1|) = log(|Sigma2|) - log(|Sigma1|)
    sgn_s1, val_s1 = jnp.linalg.slogdet(Sigma1)
    log_detSigma1 = val_s1 * sgn_s1
    sgn_s2, val_s2 = jnp.linalg.slogdet(Sigma2)
    log_detSigma2 = val_s2 * sgn_s2

    if use_chol_prec:  ## use Cholesky-factor calc of (Sigma2)^{-1}
        C = jnp.linalg.cholesky(Sigma2) ## cholesky factor matrix
        inv_C = jnp.linalg.pinv(C)
        Prec2 = jnp.matmul(inv_C.T, inv_C)
    else:
        Prec2 = jnp.linalg.pinv(Sigma2)  ## pseudo-inverse calc of (Sigma2)^{-1}

    trace_term = jnp.trace(jnp.dot(Prec2, Sigma1)) ## trace term of KL divergence
    delta_mu = mu2 - mu1
    quadratic_term = jnp.sum((jnp.matmul(delta_mu, Prec2) * delta_mu), axis=1, keepdims=True)
    #quadratic_term = jnp.matmul(jnp.matmul(delta_mu.T, Prec2), delta_mu) ## quadratic term of KL divergence
    # calc full KL divergence
    kld = ((log_detSigma2 - log_detSigma1) + quadratic_term + trace_term + quadratic_term - D) * 0.5
    return kld

@partial(jit, static_argnums=[3])
def measure_CatNLL(p, x, offset=1e-7, preserve_batch=False):
    """
    Measures the negative Categorical log likelihood (Cat.NLL).  Note: If batch is preserved, this returns a column
    vector where each row is the Cat.NLL(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities; (N x C matrix, where C is number of categories)

        x: true one-hot encoded targets; (N x C matrix, where C is number of categories)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1.0 - offset)
    loss = -(x * jnp.log(p_))
    nll = jnp.sum(loss, axis=1, keepdims=True) #/(y_true.shape[0] * 1.0)  ## CatNLL-per-datapoint
    if not preserve_batch:
        nll = jnp.mean(nll)
    return nll #tf.reduce_mean(nll)

@partial(jit, static_argnums=[2])
def measure_RMSE(mu, x, preserve_batch=False):
    """
    Measures root mean squared error (RMSE). Note: If batch is preserved, this returns a column vector where each
    row is the MSE(mu, x) for that row's datapoint. (THis is a simple wrapper/extension of the in-built MSE.)

    Args:
        mu: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    mse = measure_MSE(mu, x, preserve_batch=preserve_batch)
    return jnp.sqrt(mse) ## sqrt(MSE) is the root-mean-squared-error

@partial(jit, static_argnums=[2])
def measure_MSE(mu, x, preserve_batch=False):
    """
    Measures mean squared error (MSE), or the negative Gaussian log likelihood with variance of 1.0. Note: If batch
    is preserved, this returns a column vector where each row is the MSE(mu, x) for that row's datapoint.

    Args:
        mu: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    diff = mu - x
    se = jnp.square(diff) ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True) ## technically squared-error per data-point
    if not preserve_batch:
        mse = jnp.mean(mse) # this is proper mse
    return mse

@partial(jit, static_argnums=[2])
def measure_MAE(shift, x, preserve_batch=False):
    """
    Measures mean absolute error (MAE), or the negative Laplacian log likelihood with scale of 1.0. Note: If batch
    is preserved, this returns a column vector where each row is the MSE(mu, x) for that row's datapoint.

    Args:
        shift: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    diff = shift - x
    se = jnp.abs(diff) ## squared error
    mae = jnp.sum(se, axis=1, keepdims=True) ## technically abs-error per data-point
    if not preserve_batch:
        mae = jnp.mean(mae) # this is proper mae
    return mae

@partial(jit, static_argnums=[3])
def measure_BCE(p, x, offset=1e-7, preserve_batch=False): #1e-10
    """
    Calculates the negative Bernoulli log likelihood or binary cross entropy (BCE). Note: If batch is preserved,
    this returns a column vector where each row is the BCE(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities of shape; (N x D matrix)

        x: target binary values (data) of shape; (N x D matrix)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True) ## BCE-per-datapoint
    if not preserve_batch:
        bce = jnp.mean(bce)
    return bce
