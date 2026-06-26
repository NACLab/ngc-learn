"""
Metric and measurement routines and co-routines. These functions are useful for model-level/simulation analysis as well
as experimental inspection and probing (many of these are neuroscience-oriented measurement functions).
"""
from jax import numpy as jnp, jit
from functools import partial
from ngcsimlib import deprecated
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
def measure_breadthOfTuningCurve(responses, preserve_batch=False):
    """
    Calculates the breadth of tuning curve (BTC) of a population of neurons given full
    response window(s), across multiple trials. BTC measures the neural selectivity such that the
    sparse code distribution concentrates near zero with a heavy tail. 
    Note that, within `responses`, neural response values will include measurements of 
    characteristics such as firing rates, spike counts (burstiness measures), etc. 
    at a particular point in time/stimulus condition (each row represents a condition). 

    For a neural layer where most of the neurons fire, the activity distribution is
    more uniformly spread and BTC > 0.5. When most of the neurons do not fire,
    the firing distribution is peaked at zero and BTC < 0.5.

    Args:
        responses: tensor of shape (B x T x D), containing "evoked activity", where:
            B = number of trials (batch length; trial axis=0); 
            T = number of unique stimulus conditions / orientations (stimulus axis=1); and, 
            D = number of neurons in the population (neural index axis=2)
        preserve_batch: if True, returns (1 x D) vector of tuning widths (BTC) per neuron; 
            if False, returns scalar representing population-wide average tuning width (BTC)

    Returns: 
        a 1 x D BTC vector (one factor per neuron) OR a single average BTC across 
        the neuronal group / population
    """
    ## clean out trial noise: mean across"trials" axis (axis=0)
    clean_responses = jnp.mean(responses, axis=0, keepdims=True) ## (1xTxD)

    ## calculate mean & variance across stimulus axis (i.e., `axis=1`)
    mu = jnp.mean(clean_responses, axis=1, keepdims=True) ## (1x1xD)
    sigSqr = jnp.square(jnp.std(clean_responses, axis=1, keepdims=True)) ## (1x1xD)

    safe_mu = jnp.where(mu == 0.0, 1.0, mu)  ## check for 0/0 division for non-responsive neurons
    C = sigSqr / safe_mu
    raw_BTC = 1. / (1 + jnp.square(C)) ## raw original BTC metric

    ## NOTE: if neuron never fires for any stimulus (mu == 0),  its breadth of 
    ##       tuning is exactly 0.0 (not 1.0); thus assign it maximal sparsity
    BTC = jnp.where(mu == 0.0, 0.0, raw_BTC) ## masking check
    BTC = jnp.squeeze(BTC) ## obtain flat vector of length D (number of neurons)
    if not preserve_batch: ## calc population average
        BTC = jnp.mean(BTC)
    return BTC

@deprecated(replaced_by=measure_breadthOfTuningCurve)
def measure_breadth_TC(responses, preserve_batch=False): ## old BTC function
    """
    WARNING: this function is deprecated (renamed to `measure_breadthOfTuningCurve(.)`).
    """
    return measure_breadthOfTuningCurve(responses, preserve_batch)


@partial(jit, static_argnums=[1])
def measure_gini_index(codes, preserve_batch=True):
    """
    Calculates the gini index a group of neurons represented as vector code samples. 
    Gini index measures the sparseness of the values within each vector code, where 
    a higher index value indicates higher sparsity and a lower index value indicates a 
    lower sparsity (higher density).

    Args:
        codes: a batch of neural codes; shape is (N x D) where D is number of
            neurons in a group/cluster and N is number of samples

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar average score

    Returns:
        a N x 1 Gini index vector (one score per neuron) OR a single
        average Gini score for the whole sample/set of codes
    """
    ## Gini index
    ### values closer to 1 indicate high sparsity (sparser codes)
    ### values closer to 0 indicate lower sparsity (denser codes)
    _codes = codes + (jnp.sum(codes, axis=1, keepdims=True) <= 0.) + 1e-8
    ### note that the calculation below is faster than the mean-absolute-value 
    ### form of gini-index; below calculation requires sorting but yields a 
    ### lower-complexity calculation
    D = codes.shape[1] ## length of vector
    codes_sorted = jnp.sort(jnp.abs(_codes), axis=1) ## sort all codes w/in batch matrix
    index = jnp.arange(1, D + 1)
    term1 = jnp.sum((2 * index - D - 1) * codes_sorted, axis=1, keepdims=True)
    term2 = D * jnp.sum(codes_sorted, axis=1, keepdims=True)
    gini = term1 / term2 ## calc final ratio
    if not preserve_batch:
        gini = jnp.mean(gini) ## this is the mean gini-index
    return gini

@partial(jit, static_argnums=[2, 3])
def measure_sparsity(codes, tolerance=0., preserve_batch=True, flip_measure=False):
    """
    Calculates the sparsity (ratio) of an input matrix, assuming each row within
    this matrix is a non-negative vector. 
    
    Formally, this means we compute, per i-th row:
    
    | rho(x_i) = num_zeros(x_i) / dim(x_i)
    
    and for a global score for matrix X with N codes/rows, we measure: 
    
    | rho_mean(X) = 1/N Sum^N_{i=1} rho(x_i)
    
    where lower/closer to 0 means codes more sparse and closer to 1 means 
    codes are more dense.

    Note that this definition of sparsity aligns with Foldiak's definition of 
    the ratio of active neurons to inactive ones (assuming binary coding):

    | Foldiak, Peter. "Sparse and explicit neural coding." Principles of neural 
    | coding. CRC Press, 2013. 379-389.

    Args:
        codes: matrix (shape: N x D) of non-negative codes to measure
            sparsity of (per row)

        tolerance: lowest number to consider as "empty"/non-existent (Default: 0.)

        preserve_batch: if True, will return one score per sample (N x 1) in batch 
            (Default: True), otherwise, returns scalar average/mean score

        flip_measure: if True, will score sparsity via "1 - nzero/dim" (Default: False)

    Returns:
        sparsity measurements per code (shape: N x 1) or single score (shape: 1 x 1)
    """
    dim = codes.shape[1]
    m = (codes > tolerance).astype(jnp.float32)
    rho = jnp.sum(m, axis=1, keepdims=True)/(dim * 1.) ## per-code sparsity
    if flip_measure: ## closer to 1 = more sparse, closer to 0, more dense
        rho = 1. - rho
    if not preserve_batch:
        rho = jnp.mean(rho)
    return rho

def analyze_categorization_performance(mu, y, extract_label_indx=True): ## examines classifcation statistics
    """
    Analyzes a set of prediction matrix and target/ground-truth matrix or vector.

    Args:
        mu: prediction (design) matrix; shape is (N x C) where C is number of classes
            and N is the number of patterns examined

        y: target / ground-truth (design) matrix; shape is (N x C) OR an array
            of class integers of length N (with "extract_label_indx = True")

        extract_label_indx: wehn True, run an argmax to pull class integer indices from
            "y", assuming y is a one-hot binary encoding matrix (Default: True),
            otherwise, if False, this treats "y" is an array of class integer indices
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

@deprecated(replaced_by=analyze_categorization_performance) ## deprecated name of funct was "analyze_scores(.)"
def analyze_scores(*args, **kwargs):
    """
    WARNING: this function is deprecated (renamed to `analyze_categorization_performance(.)`).
    """
    return analyze_categorization_performance(*args, **kwargs)

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

@partial(jit, static_argnums=[1])
def measure_hoyer_sparsity(codes: jnp.ndarray, preserve_batch: bool=False) -> float:
    """
    Measures the Hoyer sparsity for a set of latent codes. 
    Hoyer sparsity lies in [0, 1], where a value of 0.0 indicates if something is dense and 
    a value of 1 indicates something is extremely sparse.

    Args:
        codes: matrix (shape: N x D) of non-negative codes to measure
            sparsity of (per row); D is flattened latent code size

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    # Flatten everything past the batch dimension
    x = jnp.reshape(codes, (codes.shape[0], -1))
    N = x.shape[1]

    l1 = jnp.sum(jnp.abs(x), axis=1)
    l2 = jnp.sqrt(jnp.sum(jnp.square(x), axis=1) + 1e-8) # epsilon to avoid division by zero

    hoyer = (jnp.sqrt(N) - (l1 / l2)) / (jnp.sqrt(N) - 1.0)
    if not preserve_batch:
        hoyer = jnp.mean(hoyer) # calc average sparsity across set/batch
    return hoyer

@partial(jit, static_argnums=[1])
def measure_excess_kurtosis(codes: jnp.ndarray, preserve_batch: bool=False) -> float:
    """
    Measures the peak and heavy-tailedness of a set of neural activation codes. Note that 
    higher values (> 0) indicate sparse, localized 'high-burst' activations.
    
    Args:
        codes: matrix (shape: N x D) of non-negative codes to measure
            sparsity of (per row)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    x = jnp.reshape(codes, (codes.shape[0], -1))
    mean = jnp.mean(x, axis=1, keepdims=True) ## 1st moment
    variance = jnp.var(x, axis=1, keepdims=True) ## 2nd moment
    
    ## 4th central moment divided by variance squared
    fourth_moment = jnp.mean(jnp.power(x - mean, 4), axis=1, keepdims=True)
    kurtosis = fourth_moment / (jnp.square(variance) + 1e-8) ## kurtosis of distribution
    excess_kurtosis = kurtosis - 3.0  ## calc "excess kurtosis" by subtracting 3
    if not preserve_batch:
        excess_kurtosis = jnp.mean(excess_kurtosis) ## calc avg excess-kurtosis over set/batch
    return excess_kurtosis 


### class conformity metrics ###

@partial(jit, static_argnums=[2, 3])
def _compute_contingency_table( ## vectorized construction of contingency matrix
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray,
        n_classes: int,
        n_clusters: int
) -> jnp.ndarray:
    ## Computes a contingency matrix table
    ## This routine expects true integer labels and predicted integer labels (1D arrays of size N)

    # Create indicator masks across all unique classes/clusters
    # find unique IDs safely up to a static maximum size (or provide num_classes)
    # n_classes = n_true = jnp.max(labels_true) + 1
    # n_clusters = n_pred = jnp.max(labels_pred) + 1

    # Broadcast to form a full one-hot lookup map
    true_mask = labels_true[:, None] == jnp.arange(n_classes)
    pred_mask = labels_pred[:, None] == jnp.arange(n_clusters)

    # Contingency matrix is the matrix product of boolean indicators
    contingency = jnp.dot(true_mask.T.astype(jnp.float32), pred_mask.astype(jnp.float32))
    return contingency


def measure_ARI(
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the adjusted random index (ARI), which measures similarity between two
    sets of indices (ground truth against a clustering's produced indices) via counting the
    pairs of data points assigned to same or different clusters (adjusted for chance). This
    measurement lies in `[0, 1]`, where `0` indicates a random labeling/assignment and `1` indicates
    perfect agreement.

    Args:
        labels_true: 1D array of shape (n_samples,) with true integer class labels.

        labels_pred: 1D array of shape (n_samples,) with predicted integer cluster labels.

    Returns:
        scalar ARI of these two sets of indices
    """
    ## Dynamically find dimensions up to a statically bounded maximum
    n_classes = int(jnp.max(labels_true) + 1)
    n_clusters = int(jnp.max(labels_pred) + 1)
    return _calc_adjusted_rand_index(labels_true, labels_pred, n_classes, n_clusters)


@partial(jit, static_argnums=[2, 3])
def _calc_adjusted_rand_index(  ## ARI
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray,
        n_classes: int,
        n_clusters: int
) -> jnp.ndarray:
    n_samples = labels_true.shape[0]
    if n_samples <= 1:
        return jnp.array(1.0)

    ## Get contingency matrix (n_classes x n_clusters)
    contingency = _compute_contingency_table(
        labels_true,
        labels_pred,
        n_classes,
        n_clusters
    )

    ## Calculate combination sums n_ijC2 = (n_ij * (n_ij - 1)) / 2
    sum_nij_c2 = jnp.sum((contingency * (contingency - 1.0)) / 2.0)

    ## Sums across margins (rows and columns)
    sum_a = jnp.sum(contingency, axis=1)
    sum_b = jnp.sum(contingency, axis=0)

    ## Margin pair combinations
    sum_a_c2 = jnp.sum((sum_a * (sum_a - 1.0)) / 2.0)
    sum_b_c2 = jnp.sum((sum_b * (sum_b - 1.0)) / 2.0)

    ## Expected index and Max index math formulas
    total_c2 = (n_samples * (n_samples - 1.0)) / 2.0
    expected_index = (sum_a_c2 * sum_b_c2) / total_c2
    max_index = (sum_a_c2 + sum_b_c2) / 2.0

    ## Prevent division by zero if everything is perfectly clustered or uniform
    denominator = max_index - expected_index
    ari = jnp.where(denominator == 0.0, 1.0, (sum_nij_c2 - expected_index) / denominator)
    return ari


def measure_FMI(
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the Fowlkes-Mallows Index (FMI), which measures similarity between two sets of
    indices - this score is the geometric mean of pair-wise recall and precision.
    This measurement lies in `[0, 1]`, where higher is better (indicating greater similarity between
    two clustering sets of identifiers).

    Args:
        labels_true: 1D array of shape (n_samples,) with true integer class labels.

        labels_pred: 1D array of shape (n_samples,) with predicted integer cluster labels.

    Returns:
        scalar FMI of these two sets of indices
    """
    ## Dynamically find dimensions up to a statically bounded maximum
    n_classes = int(jnp.max(labels_true) + 1)
    n_clusters = int(jnp.max(labels_pred) + 1)
    return _measure_fowlkes_mallows_index(labels_true, labels_pred, n_classes, n_clusters)


@partial(jit, static_argnums=[2, 3])
def _measure_fowlkes_mallows_index(  ## FMI
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray,
        n_classes: int,
        n_clusters: int
) -> jnp.ndarray:
    n_samples = labels_true.shape[0]
    # Handle edge case for single or empty samples safely
    if n_samples <= 1:
        return jnp.array(0.0, dtype=jnp.float32)

    contingency = _compute_contingency_table(labels_true, labels_pred, n_classes, n_clusters)

    ## Compute marginal sums (sums along rows and columns)
    sum_true = jnp.sum(contingency, axis=1)
    sum_pred = jnp.sum(contingency, axis=0)

    ## Calculate pairwise combinations using the matrix shortcut: nC2 = 0.5 * (sum(x^2) - N)
    # True Positives pair combinations (tk)
    tk = 0.5 * (jnp.sum(contingency ** 2) - n_samples)
    ## Total pairs clustered together in ground truth (tr)
    tr = 0.5 * (jnp.sum(sum_true ** 2) - n_samples)
    ## Total pairs clustered together in predictions (tc)
    tc = 0.5 * (jnp.sum(sum_pred ** 2) - n_samples)

    ## Compute FMI = tk / sqrt(tr * tc)
    # Prevent division by zero if there are no pair splits/matches
    denominator = jnp.sqrt(tr * tc)
    fmi = jnp.where(denominator == 0.0, 0.0, tk / denominator)
    return fmi


def measure_Vmeasure(  ## V-Measure
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray,
        beta: float = 1.0
) -> jnp.ndarray:
    """
    Calculates the V-Measure scoring metric for class conformity. This measurement compares
    predicted cluster indices ("labels_pred") against ground truth indices ("labels_true") and
    represents the harmonic mean of homogeneity (where each cluster contains only members of a single class)
    as well as completeness (where all members of a given class are assigned to the same cluster).
    This measurement (higher is better) lies in `[0,1]` where `1` indicates perfect, correct clustering.

    Args:
        labels_true: 1D array of shape (n_samples,) with true integer class labels

        labels_pred: 1D array of shape (n_samples,) with predicted integer cluster labels

         beta: Weight factor. Ratios > 1.0 favor completeness, < 1.0 favor homogeneity.

    Returns:
        scalar V-measure of these two sets of indices
    """
    ## Dynamically find dimensions up to a statically bounded maximum
    n_classes = int(jnp.max(labels_true) + 1)
    n_clusters = int(jnp.max(labels_pred) + 1)
    return _measure_v_measure_score(labels_true, labels_pred, n_classes, n_clusters, beta)


@partial(jit, static_argnums=[2, 3, 4])
def _measure_v_measure_score(  ## V-Measure
        labels_true: jnp.ndarray,
        labels_pred: jnp.ndarray,
        n_classes: int,
        n_clusters: int,
        beta: float = 1.0
) -> jnp.ndarray:
    n_samples = labels_true.shape[0]

    ## Handle edge case for single or empty samples safely
    if n_samples <= 1:
        return jnp.array(0.0, dtype=jnp.float32)

    contingency = _compute_contingency_table(labels_true, labels_pred, n_classes, n_clusters)

    ## Calculate Marginal Sums (Row and Column totals)
    sum_true = jnp.sum(contingency, axis=1)
    sum_pred = jnp.sum(contingency, axis=0)

    ## Compute Base Entropies H(True) and H(Pred)
    p_true = sum_true / n_samples
    h_true = -jnp.sum(jnp.where(p_true > 0.0, p_true * jnp.log(p_true), 0.0))

    p_pred = sum_pred / n_samples
    h_pred = -jnp.sum(jnp.where(p_pred > 0.0, p_pred * jnp.log(p_pred), 0.0))

    ## Compute Joint Entropy H(True, Pred)
    p_joint = contingency / n_samples
    h_joint = -jnp.sum(jnp.where(p_joint > 0.0, p_joint * jnp.log(p_joint), 0.0))

    ## Derive Conditional Entropies: H(True|Pred) and H(Pred|True) using identity rule
    h_true_given_pred = h_joint - h_pred
    h_pred_given_true = h_joint - h_true

    ## Compute Homogeneity (H) and Completeness (C)
    ## If base entropy is 0, the metric is perfectly satisfied (1.0)
    homogeneity = jnp.where(h_true == 0.0, 1.0, 1.0 - (h_true_given_pred / h_true))
    completeness = jnp.where(h_pred == 0.0, 1.0, 1.0 - (h_pred_given_true / h_pred))

    ## Compute Weighted Harmonic Mean (V-Measure)
    denominator = beta * homogeneity + completeness

    ## Prevent division by zero if both metrics are zero
    v_measure = jnp.where(
        denominator == 0.0,
        0.0,
        (1.0 + beta) * homogeneity * completeness / denominator
    )
    return v_measure
