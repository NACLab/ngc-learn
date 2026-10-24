# %%

from jax import random, numpy as jnp, jit
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.utils.io_utils import save_pkl, load_pkl
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.synapses.patched import PatchedSynapse
from ngclearn.components.synapses.patched.patchedSynapse import _create_multi_patch_synapses


def _get_subnet(S, mask, n_keep):
    """
    Select the sub-network, i.e., the n_keep synapses with the highest score magnitudes.

    Args:
        S: synaptic score values (at time t)

        mask: synaptic weight masking matrix (same shape as S)

        n_keep: number of synapses to keep in the sub-network

    Returns:
        a binary sub-network matrix (same shape as S)
    """
    _S = jnp.where(mask > 0., jnp.abs(S), -jnp.inf)
    threshold = jnp.sort(_S.flatten())[-n_keep]
    return jnp.where(_S >= threshold, 1., 0.)


def _calc_update(
        pre, post, W, S, mask, signVal=1., prior_type=None, prior_lmbda=0., pre_wght=1., post_wght=1.
):
    """
    Compute a tensor of adjustments to be applied to a synaptic score matrix.

    Args:
        pre: pre-synaptic statistic to drive Hebbian update

        post: post-synaptic statistic to drive Hebbian update

        W: (fixed) synaptic weight values

        S: synaptic score values (at time t)

        mask: synaptic weight masking matrix (same shape as S)

        signVal: multiplicative factor to modulate final update by (good for
            flipping the signs of a computed synaptic change matrix)

        prior_type: prior type or name (Default: None)

        prior_lmbda: prior parameter (Default: 0.0)

        pre_wght: pre-synaptic weighting term (Default: 1.)

        post_wght: post-synaptic weighting term (Default: 1.)

    Returns:
        an update/adjustment matrix (for scores)
    """

    _pre = pre * pre_wght
    _post = post * post_wght
    dS = jnp.matmul(_pre.T, _post) * W * jnp.sign(S)
    dS_reg = 0.

    if prior_type == "l2" or prior_type == "ridge":
        dS_reg = S

    if prior_type == "l1" or prior_type == "lasso":
        dS_reg = jnp.sign(S)

    if prior_type == "l1l2" or prior_type == "elastic_net":
        l1_ratio = prior_lmbda[1]
        prior_lmbda = prior_lmbda[0]
        dS_reg = jnp.sign(S) * l1_ratio + S * (1-l1_ratio)/2

    dS = dS - prior_lmbda * dS_reg

    if mask is not None:
        dS = dS * mask

    return dS * signVal


class ScorePatchedSynapse(PatchedSynapse):
    """
    A synaptic cable with fixed (random) efficacies that adjusts a score per synapse via
    a two-factor Hebbian adjustment rule; only the top-k% synapses (by score magnitude |S|)
    are used to transform signals (the edge-popup algorithm).

    | Ramanujan, Vivek, et al. "What's hidden in a randomly weighted neural network?"
    | Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - fixed value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | scores - current value matrix of synaptic scores
    | subnet - current binary mask matrix of the selected synapses (top-k% highest scores)
    | dScores - current delta matrix containing changes to be applied to synaptic scores
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        n_sub_models: The number of submodels in each layer (Default: 1 similar functionality as DenseSynapse)

        stride_shape: Stride shape of overlapping synaptic weight value matrix
            (Default: (0, 0))

        eta: global learning rate

        weight_init: a kernel to drive initialization of this synaptic cable's (fixed) values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        score_init: a kernel to drive initialization of this synaptic cable's scores
            (Default: None, which uses a fan-in uniform initialization)

        k: top k synapses kept in the sub-network (Default: 0.5)

        prior: a kernel to drive prior of this synaptic cable's scores;
            typically a tuple with 1st element as a string calling the name of
            prior to use and 2nd element as a floating point number
            calling the prior parameter lambda (Default: (None, 0.))
            currently it supports "l1" or "lasso" or "l2" or "ridge" or "l1l2" or "elastic_net".

        sign_value: multiplicative factor to apply to final score update before
            it is applied to scores; this is useful if gradient descent style
            optimization is required (as Hebbian rules typically yield
            adjustments for ascent)

        optim_type: optimization scheme to physically alter score values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

        pre_wght: pre-synaptic weighting factor (Default: 1.)

        post_wght: post-synaptic weighting factor (Default: 1.)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = (((W * subnet) * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure

        batch_size: the size of each mini batch
    """

    def __init__(
            self, name, shape, n_sub_models=1, stride_shape=(0,0), eta=0., weight_init=None, score_init=None,
            k=0.5, prior=(None, 0.), sign_value=1., optim_type="sgd", pre_wght=1., post_wght=1., p_conn=1.,
            resist_scale=1., batch_size=1, **kwargs
    ):
        super().__init__(
            name, shape, n_sub_models, stride_shape, weight_init, None, resist_scale, p_conn, batch_size, **kwargs
        )

        prior_type, prior_lmbda = prior
        self.prior_type = prior_type
        self.prior_lmbda = prior_lmbda

        ## synaptic plasticity properties and characteristics
        self.pre_wght = pre_wght
        self.post_wght = post_wght
        self.eta = eta
        self.sign_value = sign_value
        self.k = k
        n_active = jnp.sum(self.w_masks)            # w_masks  is the structural mask (block matrix mask)
        self.n_keep = int(k * n_active)             # selects among structurally available synapses

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        tmp_key, *subkeys = random.split(self.key.get(), 4)
        if score_init is None:
            score_init = DistributionGenerator.fan_in_uniform()
        scores, _, _ = _create_multi_patch_synapses(shape=shape,
                                                    n_modules=self.n_sub_models,
                                                    module_stride=self.sub_stride,
                                                    initialization_type=score_init, key=subkeys[0]
                                                    )

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, self.shape[0]))
        self.postVals = jnp.zeros((self.batch_size, self.shape[1]))
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.scores = Compartment(scores)
        self.subnet = Compartment(_get_subnet(scores, self.w_masks, self.n_keep))
        self.dScores = Compartment(jnp.zeros(self.shape))

        self.opt_params = Compartment(get_opt_init_fn(optim_type)([self.scores.get()]), auto_save=False)

    @staticmethod
    def _compute_update(mask, sign_value, prior_type, prior_lmbda, pre_wght, post_wght, pre, post, weights, scores):
        ## calculate synaptic score update values
        dS = _calc_update(
            pre, post, weights, scores, mask, signVal=sign_value, prior_type=prior_type, prior_lmbda=prior_lmbda,
            pre_wght=pre_wght, post_wght=post_wght
        )
        return dS

    def save(self, directory: str):
        super().save(directory)
        # Also save the optimizer parameters
        save_pkl(directory, self.name + "_opt_params", self.opt_params.get())

    def load(self, directory: str):
        super().load(directory)
        # load the optimizer parameters in a custom way
        self.opt_params.set(load_pkl(directory, self.name + "_opt_params"))

    @compilable
    def advance_state(self):
        # Get the variables
        weights = self.weights.get() * self.subnet.get()
        biases = self.biases.get()


        ################### inputs  >>  W  >> outputs = (inputs @ W)
        inputs = self.inputs.get()
        ## Compute (inputs @ W)
        outputs = (jnp.matmul(inputs, weights) * self.Rscale) + biases
        ## Update outputs compartment
        self.outputs.set(outputs)


        ###################    post_in >>  W.T  >> pre_out = (post_in @ W.T)
        post_in = self.post_in.get()
        ## Compute (post_in @ W.T)
        pre_out = jnp.matmul(post_in, weights.T)
        ## Update pre_out compartment
        self.pre_out.set(pre_out)


        ################### project_input >>  W  >> project_output = (project_input @ W)
        project_input = self.project_input.get()
        ## Compute (project_input @ W)
        project_output = (jnp.matmul(project_input, weights) * self.Rscale) + biases
        ## Update project_output compartment
        self.project_output.set(project_output)


    @compilable
    def evolve(self):
        # Get the variables
        pre = self.pre.get()
        post = self.post.get()
        weights = self.weights.get()
        scores = self.scores.get()
        opt_params = self.opt_params.get()

        ## calculate synaptic score update values
        dScores = ScorePatchedSynapse._compute_update(
            self.w_masks, self.sign_value, self.prior_type, self.prior_lmbda, self.pre_wght, self.post_wght,
            pre, post, weights, scores
        )
        ## conduct a step of optimization - get newly evolved synaptic score value matrix
        opt_params, [scores] = self.opt(opt_params, [scores], [dScores])
        ## Find the Sub-network given new scores
        subnet = _get_subnet(scores, self.w_masks, self.n_keep)

        # Update compartments
        self.opt_params.set(opt_params)
        self.scores.set(scores)
        self.subnet.set(subnet)
        self.dScores.set(dScores)

    @compilable
    def reset(self):  ## closed, no-batch argument reset
        ## write reset command to call inner batched_reset command
        self.batched_reset(batch_size=self.batch_size) ## arg = batch_size data-member

    @compilable
    def batched_reset(self, batch_size): ## open, batch argument reset
        preVals = jnp.zeros((batch_size, self.shape[0]))
        postVals = jnp.zeros((batch_size, self.shape[1]))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(preVals)  # inputs
        self.outputs.set(postVals)               # outputs
        self.project_input.set(preVals)          # project_input
        self.project_output.set(postVals)        # project_output
        self.post_in.set(postVals)               # post_in
        self.pre_out.set(preVals)                # pre_out
        self.pre.set(preVals)                    # pre
        self.post.set(postVals)                  # post
        self.dScores.set(jnp.zeros(self.shape))  # dS

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "ScorePatchedSynapse - performs a synaptic transformation of inputs "
                            "to produce output signals through the top-k% (fixed, random) synapses "
                            "ranked by score; scores are adjusted via two-term/factor Hebbian adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "project_input": "Takes in external input signal values",
                 "post_in": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"weights": "Fixed synapse efficacy/strength parameter values",
                 "scores": "Synapse (popup) score parameter values",
                 "subnet": "Binary selection matrix of the top-k% synapses",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dScores": "Synaptic score value adjustment matrix produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation",
                 "project_output": "Output of synaptic transformation",
                 "pre_out": "Output of synaptic transformation",
                 },
        }
        hyperparams = {
            "shape": "Overall shape of synaptic weight value matrix; number inputs x number outputs",
            "n_sub_models": "The number of submodels in each layer",
            "stride_shape": "Stride shape of overlapping synaptic weight value matrix",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for (fixed) synaptic weight (W) values",
            "score_init": "Initialization conditions for synaptic score (S) values",
            "k": "Fraction of synapses kept in the sub-network",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "pre_wght": "Pre-synaptic weighting coefficient (q_pre)",
            "post_wght": "Post-synaptic weighting coefficient (q_post)",
            "prior": "prior name and value for synaptic score updating prior",
            "optim_type": "Choice of optimizer to adjust synaptic scores"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [((W * M) * Rscale) * inputs] + b ; M = 1 if |S_{ij}| in top-k% else 0;"
                            "dS_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] * W_{ij} * sign(S_{ij}) - g(S_{ij}) * prior_lmbda",
                "hyperparameters": hyperparams}
        return info
