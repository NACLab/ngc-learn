import jax.numpy as jnp
from jax import random, jit

from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats
from ngcsimlib import deprecate_args
#from ngclearn.utils.io_utils import save_pkl, load_pkl

class GerstnerHebbianSynapse(DenseSynapse):
    """
    A synapse component that implements Gerstner's general Hebbian
    learning (Taylor) expansion (Equation 3 from Gerstner & Kistler, 2002).

    Note that this synaptic update model can recover several classical forms
    of Hebbian-like update rules, including the covariance rule.

    There are other higher-order terms possible, i.e., Theta(xy), such as
    x * y2 and y x^2, etc.

    | c2_corr > 0 and c0 = c1_pre = c1_post = 0 => Hebbian update
    | c2_corr < 0 and c0 = c1_pre = c1_post = 0 => anti-Hebbian update
    | c2_corr = 1 and c1_pre = -x_theta < 0

    | References: 
    | Gerstner, W. and Kistler, W.M., 2002. Mathematical formulations of Hebbian 
    | learning. Biological cybernetics, 87(5), pp.404-415.

    """
    def __init__(
        self,
        name,
        shape, ## (post_dim, pre_dim)
        eta=0.01, ## global step-size
        coeffs=None, ## these configure which kind of Hebb learning is done
        weight_init=None,
        p_conn=1.,
        resist_scale=1.,
        sign_value=1.,
        batch_size=1,
        **kwargs
    ):
        bias_init = None ## NOTE: no biases are included in Gerster's formulation
        super().__init__(
            name,
            shape=shape,
            weight_init=weight_init,
            bias_init=bias_init,
            resist_scale=resist_scale,
            p_conn=p_conn,
            batch_size=batch_size,
            **kwargs
        )
        ## general Hebbian meta-parameters
        self.eta = eta
        self.sign_value = sign_value

        ## Gerstner and Kisler's expansion coefficients (c0, c1_pre, c1_post, c2_corr)
        if coeffs is None: ## Default to standard bilinear Hebb
            self.coeffs = {
                'c0': 0., 'c1_pre': 0., 'c1_post': 0., 'c2_corr': 1.0
            }
        else:
            self.coeffs = coeffs
        self.c0 = self.coeffs['c0']
        self.c1_pre = self.coeffs['c1_pre']
        self.c1_post = self.coeffs['c1_post']
        self.c2_corr = self.coeffs['c2_corr']

        ## set up relevant compartments
        self.pre = Compartment(jnp.zeros((1, shape[1])))
        self.post = Compartment(jnp.zeros((1, shape[0])))
        self.dWeights = Compartment(jnp.zeros(shape))

    @compilable
    def evolve(self, **kwargs): ## perform update via Gerstner's general expansion
        ## retrieve current compartment state values
        W = self.weights.get()
        x = self.pre.get()  ## pre-synaptic activity (batch, pre_dim)
        y = self.post.get() ## post-synaptic activity (batch, post_dim)
        batch_size = self.batch_size

        ## calculate bilinear Term (c2), i.e., correlation matrix
        ### (pre_dim, batch) @ (batch, post_dim) -> (pre_dim, post_dim)
        dW_corr = jnp.matmul(x.T, y) * (1./batch_size)
        ## linear pre-synaptic term (c1_pre)
        ### get mean over batch then broadcast to match weight matrix
        dW_pre = jnp.sum(x, axis=0, keepdims=True).T * (1./batch_size)
        ## linear post-synaptic term (c1_post), mean over post-syn values
        dW_post = jnp.sum(y, axis=0, keepdims=True) * (1./batch_size) 

        ## apply Taylor expansion from Equation 3 (Gerstner and Kistler)
        dW = (self.c0 * W +  ## synaptic decay
              self.c1_pre * dW_pre +  ## bilinear term
              self.c1_post * dW_post +  ## pre-synaptic gating term
              self.c2_corr * dW_corr  ## post-synpatic gating term
        )
        self.dWeights.set(dW)

        ## perform a step of Hebbian ascent
        W = W + self.eta * dW ## update synaptic efficacies
        self.weights.set(W)

    @compilable
    def reset(self, **kwargs): ## clear compartment values
        self.pre.set( jnp.zeros((self.batch_size, self.shape[1])) )
        self.post.set( jnp.zeros((self.batch_size, self.shape[0])) )
        self.dWeights.set(self.dWeights.get() * 0)

