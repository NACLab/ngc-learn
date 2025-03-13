import jax
import numpy as np
from ngclearn.utils.analysis.probe import Probe
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from ngclearn.utils.optim import adam
from jax import jit, random, numpy as jnp, lax, nn
from functools import partial as bind

def masked_fill(x: jax.Array, mask: jax.Array, value=0) -> jax.Array:
    """
    Return an output with masked condition, with non-masked value
    be the other value

    Args:
        x (jax.Array): _description_
        mask (jax.Array): _description_
        value (int, optional): _description_. Defaults to 0.

    Returns:
        jax.Array: _description_
    """
    return jnp.where(mask, jnp.broadcast_to(value, x.shape), x)

@bind(jax.jit, static_argnums=[5, 6])
def cross_attention(dkey, params: tuple, x1: jax.Array, x2: jax.Array, mask: jax.Array, n_heads: int=8, dropout_rate: float=0.0) -> jax.Array:
    """
    Run cross-attention function given a list of parameters and two sequences (x1 and x2).
    The function takes in a query sequence x1 and a key-value sequence x2, and returns an output of the same shape as x1.
    T is the length of the query sequence, and S is the length of the key-value sequence.
    Dq is the dimension of the query sequence, and Dkv is the dimension of the key-value sequence.
    H is the number of attention heads.

    Args:
        dkey: JAX key to trigger any internal noise (drop-out)

        params (tuple): tuple of parameters

        x1 (jax.Array): query sequence. Shape: (B, T, Dq)

        x2 (jax.Array): key-value sequence. Shape: (B, S, Dkv)

        mask (jax.Array): mask tensor. Shape: (B, T, S)

        n_heads (int, optional): number of attention heads. Defaults to 8.

        dropout_rate (float, optional): dropout rate. Defaults to 0.0.

    Returns:
        jax.Array: output of cross-attention
    """
    B, T, Dq = x1.shape # The original shape
    _, S, Dkv = x2.shape
    # in here we attend x2 to x1
    Wq, bq, Wk, bk, Wv, bv, Wout, bout = params
    # projection
    q = x1 @ Wq + bq # normal linear transformation (B, T, D)
    k = x2 @ Wk + bk # normal linear transformation (B, S, D)
    v = x2 @ Wv + bv # normal linear transformation (B, S, D)
    hidden = q.shape[-1]
    _hidden = hidden // n_heads
    q = q.reshape((B, T, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    k = k.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    v = v.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(_hidden) # Q @ KT / ||d||; d = D // n_heads
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
        _mask = mask.reshape((B, 1, Tq, Tk)) # 'b tq tk -> b 1 tq tk'
        score = masked_fill(score, _mask, value=-jnp.inf) # basically masking out all must-unattended values
    score = jax.nn.softmax(score, axis=-1) # (B, H, T, S)
    score = score.astype(q.dtype) # (B, H, T, S)
    if dropout_rate > 0.:
        score, _ = drop_out(dkey, score, rate=dropout_rate) ## NOTE: normally you apply dropout here
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v) # (B, T, H, E)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, T, -1)) # (B, T, H, E) => (B, T, D)
    return attention @ Wout + bout # (B, T, Dq)

@bind(jax.jit, static_argnums=[4, 5, 6, 7, 8])
def run_attention_probe(
        dkey, params, encodings, mask, n_heads: int, dropout: float = 0.0, use_LN=False, use_LN_input=False,
        use_softmax=True
):
    """
    Runs full nonlinear attentive probe on input encodings (typically embedding vectors produced by some other model). 

    Args:
        dkey: JAX key for any internal noise to be applied

        params: parameters tuple/list of probe

        encodings: input encoding vectors/data

        mask: optional mask to be applied to internal cross-attention

        n_heads: number of attention heads

        dropout: if >0, triggers drop-out applied internally to cross-attention

        use_LN: use layer normalization?

        use_LN_input: use layer normalization on input encodings?

        use_softmax: should softmax be applied to output of attention probe? (useful for classification) 

    Returns:
        output scores/probabilities, cross-attention (hidden) features
    """
    # Two separate dkeys for each dropout in two cross attention
    dkey1, dkey2 = random.split(dkey, 2)
    # encoded_image_feature: (B, hw, dim)
    #learnable_query, *_params) = params
    learnable_query, Wq, bq, Wk, bk, Wv, bv, Wout, bout,\
        Wqs, bqs, Wks, bks, Wvs, bvs, Wouts, bouts, Wlnattn_mu,\
        Wlnattn_scale, Whid1, bhid1, Wln_mu1, Wln_scale1, Whid2,\
        bhid2, Wln_mu2, Wln_scale2, Whid3, bhid3, Wln_mu3, Wln_scale3,\
        Wy, by, ln_in_mu, ln_in_scale, ln_in_mu2, ln_in_scale2 = params
    cross_attn_params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
    if use_LN_input:
        learnable_query = layer_normalize(learnable_query, ln_in_mu, ln_in_scale)
        encodings = layer_normalize(encodings, ln_in_mu2, ln_in_scale2)
    features = cross_attention(dkey1, cross_attn_params, learnable_query, encodings, mask, n_heads, dropout)
    # Perform a single self-attention block here
    # Self-Attention
    self_attn_params = (Wqs, bqs, Wks, bks, Wvs, bvs, Wouts, bouts)
    skip = features
    if use_LN:
        features = layer_normalize(features, Wlnattn_mu, Wlnattn_scale)
    features = cross_attention(dkey2, self_attn_params, features, features, None, n_heads, dropout)
    features = features + skip
    features = features[:, 0]  # (B, 1, dim) => (B, dim)
    # MLP
    skip = features
    if use_LN: ## normalize hidden layer output of probe predictor
        features = layer_normalize(features, Wln_mu1, Wln_scale1)
    features = jnp.matmul((features), Whid1) + bhid1
    features = gelu(features)
    if use_LN: ## normalize hidden layer output of probe predictor
        features = layer_normalize(features, Wln_mu2, Wln_scale2)
    features = jnp.matmul((features), Whid2) + bhid2
    features = gelu(features)
    if use_LN: ## normalize hidden layer output of probe predictor
        features = layer_normalize(features, Wln_mu3, Wln_scale3)
    features = jnp.matmul((features), Whid3) + bhid3
    features = features + skip
    outs = jnp.matmul(features, Wy) + by
    if use_softmax: ## apply softmax output nonlinearity
        # NOTE: Viet: please check the softmax function, it might potentially
        # cause the gradient to be nan since there is a potential division by zero
        outs = jax.nn.softmax(outs)
    return outs, features

@bind(jax.jit, static_argnums=[5, 6, 7, 8, 9])
def eval_attention_probe(dkey, params, encodings, labels, mask, n_heads: int, dropout: float = 0.0, use_LN=False, use_LN_input=False, use_softmax=True):
    """
    Runs and evaluates the nonlinear attentive probe given a paired set of encoding vectors and externally assigned 
    labels/regression targets.

    Args:
        dkey: JAX key to trigger any internal noise (as in drop-out)

        params: parameters tuple/list of probe

        encodings: input encoding vectors/data

        labels: output target values (e.g., labels, regression target vectors)

        mask: optional mask to be applied to internal cross-attention

        n_heads: number of attention heads

        dropout: if >0, triggers drop-out applied internally to cross-attention

        use_LN: use layer normalization?

        use_softmax: should softmax be applied to output of attention probe? (useful for classification) 

    Returns:
        current loss value, output scores/probabilities
    """
    # encodings: (B, hw, dim)
    outs, _ = run_attention_probe(dkey, params, encodings, mask, n_heads, dropout, use_LN, use_LN_input, use_softmax)
    if use_softmax: ## Multinoulli log likelihood for 1-of-K predictions
        L = -jnp.mean(jnp.sum(jnp.log(outs.clip(min=1e-5)) * labels, axis=1, keepdims=True))
    else: ## MSE for real-valued outputs
        L = jnp.mean(jnp.sum(jnp.square(outs - labels), axis=1, keepdims=True))
    return L, outs #, features

class AttentiveProbe(Probe):
    """
    This implements a nonlinear attentive probe, which is useful for evaluating the quality of 
    encodings/embeddings in light of some superivsory downstream data (e.g., label one-hot 
    encodings or real-valued vector regression targets).

    Args:
        dkey: init seed key

        source_seq_length: length of input sequence (e.g., height x width of the image feature)

        input_dim: input dimensionality of probe

        out_dim: output dimensionality of probe

        num_heads: number of cross-attention heads

        head_dim: output dimensionality of each cross-attention head

        target_seq_length: to pool, we set it at one (or map the source sequence to the target sequence of length 1)

        learnable_query_dim: target sequence dim (output dimension of cross-attention portion of probe)

        batch_size: size of batches to process per internal call to update (or process)

        hid_dim: dimensionality of hidden layer(s) of MLP portion of probe

        use_LN: should layer normalization be used within MLP portions of probe or not?

        use_softmax: should a softmax be applied to output of probe or not?

    """
    def __init__(
            self, dkey, source_seq_length, input_dim, out_dim, num_heads=8, attn_dim=64,
            target_seq_length=1, learnable_query_dim=32, batch_size=1, hid_dim=32,
            use_LN=True, use_LN_input=False, use_softmax=True, dropout=0.5, eta=0.0002,
            eta_decay=0.0, min_eta=1e-5, **kwargs
    ):
        super().__init__(dkey, batch_size, **kwargs)
        assert attn_dim % num_heads == 0, f"`attn_dim` must be divisible by `num_heads`. Got {attn_dim} and {num_heads}."
        assert learnable_query_dim % num_heads == 0, f"`learnable_query_dim` must be divisible by `num_heads`. Got {learnable_query_dim} and {num_heads}."
        self.dkey, *subkeys = random.split(self.dkey, 26)
        self.num_heads = num_heads
        self.source_seq_length = source_seq_length
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_softmax = use_softmax
        self.use_LN = use_LN
        self.use_LN_input = use_LN_input
        self.dropout = dropout

        sigma = 0.02
        ## cross-attention parameters
        Wq = random.normal(subkeys[0], (learnable_query_dim, attn_dim)) * sigma
        bq = random.normal(subkeys[1], (1, attn_dim)) * sigma
        Wk = random.normal(subkeys[2], (input_dim, attn_dim)) * sigma
        bk = random.normal(subkeys[3], (1, attn_dim)) * sigma
        Wv = random.normal(subkeys[4], (input_dim, attn_dim)) * sigma
        bv = random.normal(subkeys[5], (1, attn_dim)) * sigma
        Wout = random.normal(subkeys[6], (attn_dim, learnable_query_dim)) * sigma
        bout = random.normal(subkeys[7], (1, learnable_query_dim)) * sigma
        cross_attn_params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
        Wqs = random.normal(subkeys[8], (learnable_query_dim, learnable_query_dim)) * sigma
        bqs = random.normal(subkeys[9], (1, learnable_query_dim)) * sigma
        Wks = random.normal(subkeys[10], (learnable_query_dim, learnable_query_dim)) * sigma
        bks = random.normal(subkeys[11], (1, learnable_query_dim)) * sigma
        Wvs = random.normal(subkeys[12], (learnable_query_dim, learnable_query_dim)) * sigma
        bvs = random.normal(subkeys[13], (1, learnable_query_dim)) * sigma
        Wouts = random.normal(subkeys[14], (learnable_query_dim, learnable_query_dim)) * sigma
        bouts = random.normal(subkeys[15], (1, learnable_query_dim)) * sigma
        Wlnattn_mu = jnp.zeros((1, learnable_query_dim)) ## LN parameter (applied to output of attention)
        Wlnattn_scale = jnp.ones((1, learnable_query_dim)) ## LN parameter (applied to output of attention)
        self_attn_params = (Wqs, bqs, Wks, bks, Wvs, bvs, Wouts, bouts, Wlnattn_mu, Wlnattn_scale)
        learnable_query = jnp.zeros((batch_size, 1, learnable_query_dim))  # (B, T, D)
        self.mask = np.zeros((self.batch_size, target_seq_length, source_seq_length)).astype(bool) ## mask tensor
        self.dev_mask = np.zeros((self.dev_batch_size, target_seq_length, source_seq_length)).astype(bool)
        ## MLP parameters
        Whid1 = random.normal(subkeys[16], (learnable_query_dim, learnable_query_dim)) * sigma
        bhid1 = random.normal(subkeys[17], (1, learnable_query_dim)) * sigma
        Wln_mu1 = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        Wln_scale1 = jnp.ones((1, learnable_query_dim)) ## LN parameter
        Whid2 = random.normal(subkeys[18], (learnable_query_dim, learnable_query_dim * 4)) * sigma
        bhid2 = random.normal(subkeys[19], (1, learnable_query_dim * 4)) * sigma
        Wln_mu2 = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        Wln_scale2 = jnp.ones((1, learnable_query_dim)) ## LN parameter
        Whid3 = random.normal(subkeys[20], (learnable_query_dim * 4, learnable_query_dim)) * sigma
        bhid3 = random.normal(subkeys[21], (1, learnable_query_dim)) * sigma
        Wln_mu3 = jnp.zeros((1, learnable_query_dim * 4)) ## LN parameter
        Wln_scale3 = jnp.ones((1, learnable_query_dim * 4)) ## LN parameter
        Wy = random.normal(subkeys[22], (learnable_query_dim, out_dim)) * sigma
        by = random.normal(subkeys[23], (1, out_dim)) * sigma
        mlp_params = (Whid1, bhid1, Wln_mu1, Wln_scale1, Whid2, bhid2, Wln_mu2, Wln_scale2, Whid3, bhid3, Wln_mu3, Wln_scale3, Wy, by)
        # Finally, define ln for the input to the attention
        ln_in_mu = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        ln_in_scale = jnp.ones((1, learnable_query_dim)) ## LN parameter
        ln_in_mu2 = jnp.zeros((1, input_dim)) ## LN parameter
        ln_in_scale2 = jnp.ones((1, input_dim)) ## LN parameter
        ln_in_params = (ln_in_mu, ln_in_scale, ln_in_mu2, ln_in_scale2)
        self.probe_params = (learnable_query, *cross_attn_params, *self_attn_params, *mlp_params, *ln_in_params)

        ## set up gradient calculator
        self.grad_fx = jax.value_and_grad(eval_attention_probe, argnums=1, has_aux=True) #, allow_int=True)
        ## set up update rule/optimizer
        self.optim_params = adam.adam_init(self.probe_params)
        # Learning rate scheduling
        self.eta = eta #0.001
        self.eta_decay = eta_decay
        self.min_eta = min_eta

        # Finally, the dkey for the noise_key
        self.noise_key = subkeys[24]

    def process(self, embeddings, dkey=None):
        # noise_key = None
        noise_key = self.noise_key
        if dkey is not None:
            dkey, *subkeys = random.split(dkey, 2)
            noise_key = subkeys[0]
        outs, feats = run_attention_probe(
            noise_key, self.probe_params, embeddings, self.dev_mask, self.num_heads, 0.0,
            use_LN=self.use_LN, use_LN_input=self.use_LN_input, use_softmax=self.use_softmax
        )
        return outs

    def update(self, embeddings, labels, dkey=None):
        # noise_key = None
        noise_key = self.noise_key
        if dkey is not None:
            dkey, *subkeys = random.split(dkey, 2)
            noise_key = subkeys[0]
        outputs, grads = self.grad_fx(
            noise_key, self.probe_params, embeddings, labels, self.mask, self.num_heads, dropout=self.dropout,
            use_LN=self.use_LN, use_LN_input=self.use_LN_input, use_softmax=self.use_softmax
        )
        loss, predictions = outputs
        ## adjust parameters of probe
        self.optim_params, self.probe_params = adam.adam_step(
            self.optim_params, self.probe_params, grads, eta=self.eta
        )

        self.eta = max(self.min_eta, self.eta - self.eta_decay * self.eta)
        return loss, predictions

