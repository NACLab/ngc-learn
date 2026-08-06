# %%

from jax import numpy as jnp, random, jit

from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function, threshold_soft, \
                                       threshold_cauchy
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2, step_rk4
from ngclearn.components.synapses.patched import HebbianPatchedSynapse, StaticPatchedSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngcsimlib.logger import info


def _dfz_internal_laplace(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = jnp.sign(z) ## d/dx of Laplace is signum
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_cauchy(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = (z * 2)/(1. + jnp.square(z))
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_exp(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = jnp.exp(-jnp.square(z)) * z * 2
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_gaussian(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = z # * 2 ## Default: assume Gaussian
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

# @jit
def _modulate(j, dfx_val):
    """
    Apply a signal modulator to j (typically of the form of a derivative/dampening function)

    Args:
        j: current/stimulus value to modulate

        dfx_val: modulator signal

    Returns:
        modulated j value
    """
    return j * dfx_val

# @partial(jit, static_argnames=["integType", "priorType"])
def _run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., integType=0, priorType=0):
    """
    Runs leaky rate-coded state dynamics one step in time.

    Args:
        dt: integration time constant

        j: input (bottom-up) electrical/stimulus current

        j_td: modulatory (top-down) electrical/stimulus pressure

        z: current value of membrane/state

        tau_m: membrane/state time constant

        leak_gamma: strength of leak to apply to membrane/state

        integType: integration type to use (0 --> Euler/RK1, 1 --> Midpoint/RK2, 2 --> RK4)

        priorType: scale-shift prior distribution to impose over neural dynamics

    Returns:
        New value of membrane/state for next time step
    """
    _dfz_fns = {
        0: lambda t, z, params: _dfz_internal_gaussian(z, *params),
        1: lambda t, z, params: _dfz_internal_laplace(z, *params),
        2: lambda t, z, params: _dfz_internal_cauchy(z, *params),
        3: lambda t, z, params: _dfz_internal_exp(z, *params),
    }
    _dfz_fn = _dfz_fns.get(priorType, _dfz_internal_gaussian)
    _step_fns = {
        0: step_euler,
        1: step_rk2,
        2: step_rk4,
    }
    _step_fn = _step_fns.get(integType, step_euler)
    params = (j, j_td, tau_m, leak_gamma)
    _, _z = _step_fn(0., z, _dfz_fn, dt, params)
    return _z

# @jit
def _run_cell_stateless(j):
    """
    A simplification of running a stateless set of dynamics over j (an identity
    functional form of dynamics).

    Args:
        j: stimulus to do nothing to

    Returns:
        the stimulus
    """
    return j + 0

class RateCell(JaxComponent): ## Rate-coded/real-valued cell
    """
    A non-spiking cell driven by the gradient dynamics of neural generative
    coding-driven predictive processing.

    The specific differential equation that characterizes this cell
    is (for adjusting v, given current j, over time) is:

    | tau_m * dz/dt = lambda * prior(z) + (j + j_td + j_lat(if use_lateral==True))
    | where j is the set of general incoming input signals (e.g., message-passed signals)
    | j_lat (optional) lateral pressure via lateral synapse
    | and j_td is taken to be the set of top-down pressure signals

    | --- Cell Input Compartments: ---
    | j - input pressure (takes in external signals)
    | j_td - input/top-down pressure input (takes in external signals)
    | --- Cell State Compartments ---
    | z - rate activity
    | --- Cell Output Compartments: ---
    | zF - post-activation function activity, i.e., fx(z)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane/state time constant (milliseconds)

        prior: a kernel for specifying the type of centered scale-shift distribution
            to impose over neuronal dynamics, applied to each neuron or
            dimension within this component (Default: ("gaussian", 0)); this is
            a tuple with 1st element containing a string name of the distribution
            one wants to use while the second value is a `leak rate` scalar
            that controls the influence/weighting that this distribution
            has on the dynamics; for example, ("laplacian, 0.001") means that a
            centered laplacian distribution scaled by `0.001` will be injected
            into this cell's dynamics ODE each step of simulated time

            :Note: supported scale-shift distributions include "laplacian",
                "cauchy", "exp", and "gaussian"

        act_fx: string name of activation function/nonlinearity to use

        output_scale: factor to multiply output of nonlinearity of this cell by (Default: 1.)

        threshold: the kind of iterative thresholding function to place over neuronal dynamics

        batch_size: the size of each mini batch

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)

        resist_scale: a scaling factor applied to incoming pressure `j` (default: 1)

        use_lateral: determines wheter the RateCell uses lateral pressure during inference (Default: False)

        n_patch:  the number of local patches in each layer (equivalent to n_sub_models in PatchedSynapse) (Default: 1)

        adaptive_lateral: if True, lateral synaptic is adaptive using HebbianPatchedSynapse component,
                          if False, lateral synaptic is a static using StaticPatchedSynapse component
                          (Default: False)

        Wl_eta: learning rate for the lateral synapse update (Default: 0.001)

        Wl_opt_type: optimizer used by the lateral synapse (Default: "sgd")

        exc_inh: lateral synaptic initialization values: (Default: (0., 0.))

        fx_lat: string name of activation function/nonlinearity to use for lateral pressure (Default: same as act_fx)

        Wl_prior: a kernel to drive prior over lateral synaptic cable's values (Default: (None, 0.))

        Wl_bound: maximum value to enforce over newly computed lateral synaptic values (Default: 0.)

    """

    def __init__(
            self, name, n_units, tau_m, prior=("gaussian", 0.), act_fx="identity", output_scale=1., threshold=("none", 0.),
            integration_type="euler",
            batch_size=1,
            resist_scale=1.,
            shape=None,
            is_stateful=True,
            use_lateral = False,
            adaptive_lateral = False,
            exc_inh=(0., 0.),
            n_patch=1,
            fx_lat=None,
            Wl_eta=0.,
            Wl_opt_type="sgd",
            Wl_prior=(None, 0.),
            **kwargs):
        jax_comp_kwargs = {k: v for k, v in kwargs.items() if k in ('key',)}
        this_class_kwargs = {k: v for k, v in kwargs.items() if k in ('omega_0',)}
        super().__init__(name, **jax_comp_kwargs)

        ## membrane parameter setup (affects ODE integration)
        self.output_scale = output_scale
        self.tau_m = tau_m ## membrane time constant -- setting to 0 triggers "stateless" mode
        self.is_stateful = is_stateful
        if isinstance(tau_m, float):
            if tau_m <= 0: ## trigger stateless mode
                self.is_stateful = False
        priorType, leakRate = prior
        priorTypeDict = {
            "gaussian": 0,
            "laplacian": 1,
            "cauchy": 2,
            "exp": 3
        }
        self.priorType = priorTypeDict.get(priorType, 0)
        self.priorLeakRate = leakRate ## degree to which rate neurons leak (according to prior)
        thresholdType, thr_lmbda = threshold
        self.thresholdType = thresholdType ## type of thresholding function to use
        self.thr_lmbda = thr_lmbda ## scale to drive thresholding dynamics
        self.resist_scale = resist_scale ## a "resistance" scaling factor

        self.use_lateral = use_lateral
        self.adaptive_lateral = adaptive_lateral
        self.Wl_eta = Wl_eta

        ## integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Layer size setup
        _shape = (batch_size, n_units) ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,) ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2]) ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.n_patch = n_patch
        self.dim_patch = n_units // n_patch
        self.batch_size = batch_size

        omega_0 = None
        if act_fx == "sine":
            omega_0 = this_class_kwargs["omega_0"]
        self.fx, self.dfx = create_function(fun_name=act_fx, args=omega_0)
        if use_lateral:
            if fx_lat is None:
                self.fx_lat, self.dfx_lat = self.fx, self.dfx
            else:
                self.fx_lat, self.dfx_lat = create_function(fun_name=fx_lat)

        ## ══════════════════ lateral synapse ══════════════════
        self.Wl = None
        if use_lateral:
            exc, inh = exc_inh
            d = n_units // n_patch

            if adaptive_lateral:
                self.Wl = HebbianPatchedSynapse(f"Wl_{name}", shape=(n_units, n_units),
                                                n_sub_models=n_patch,
                                                weight_init=dist.diag_offdiag(on_diag=exc, off_diag=inh),
                                                eta=Wl_eta,
                                                prior=Wl_prior,
                                                sign_value=-1.,
                                                optim_type=Wl_opt_type,
                                                batch_size=batch_size,
                                                )
            else:
                self.Wl = StaticPatchedSynapse(f"Wl_{name}", shape=(n_units, n_units),
                                               n_sub_models=n_patch,
                                               weight_init=dist.diag_offdiag(on_diag=exc, off_diag=inh),
                                               batch_size=batch_size,
                                               )

        ## ════════════════════════════════════════════════════════════════════════════════════
        ## compartments (state of the cell & parameters will be updated through stateless calls)
        restVals = jnp.zeros(_shape)

        self.j = Compartment(restVals, display_name="Input Stimulus Current", units="mA") # electrical current
        self.zF = Compartment(restVals, display_name="Transformed Rate Activity") # rate-coded output - activity
        self.j_td = Compartment(restVals, display_name="Modulatory Stimulus Current", units="mA") # top-down electrical current - pressure
        self.z = Compartment(restVals, display_name="Rate Activity", units="mA") # rate activity
        ## lateral compartments
        if use_lateral:
            self.j_lat = Compartment(restVals, display_name="Lateral Stimulus Current", units="mA")  # lateral current
            self.zF_lat = Compartment(restVals, display_name="Lateral Rate Activity")  # fx_lat(z)

    def compile(self):
        if self.use_lateral:
            self.zF_lat >> self.Wl.inputs
            self.Wl.outputs >> self.j_lat

            if self.adaptive_lateral:
                self.zF_lat >> self.Wl.pre
                self.zF_lat >> self.Wl.post
        super().compile()

    @compilable
    def advance_state(self, dt):
        # Get the compartment values
        j = self.j.get()
        j_td = self.j_td.get()
        z = self.z.get()

        if self.is_stateful:
            ## ════════════════════════════════════════════════════════════════
            ### run a step of integration over neuronal dynamics
            ## Note:
            ## self.pressure <-- "top-down" expectation / contextual pressure
            ## ══════════════════════════
            ## self.current <-- "bottom-up" data-dependent signal
            dfx_val = self.dfx(z)
            j = _modulate(j, dfx_val)
            j = j * self.resist_scale

            ## ══════════════════════════
            ## Calculate lateral pressure
            if self.use_lateral:
                self.Wl.advance_state()

                j_lat = self.j_lat.get()
                dfx_lat = self.dfx_lat(z)
                j_lat = _modulate(j_lat, dfx_lat)
                j_lat = j_lat * self.resist_scale
            else:
                j_lat = jnp.zeros_like(z)

            ## ═════════════════════════════════════════════════
            ## Run a single integration step for a stateful cell
            tmp_z = _run_cell(dt,
                              j=(j + j_lat),
                              j_td=j_td,
                              z=z,
                              tau_m=self.tau_m,
                              leak_gamma=self.priorLeakRate,
                              integType=self.intgFlag,
                              priorType=self.priorType
                              )

            ## apply optional thresholding sub-dynamics
            if self.thresholdType == "soft_threshold":
                tmp_z = threshold_soft(tmp_z, self.thr_lmbda)
            elif self.thresholdType == "cauchy_threshold":
                tmp_z = threshold_cauchy(tmp_z, self.thr_lmbda)
            z = tmp_z  ## pre-activation function value(s)
        else:
            ## ════════════════════════════════════════════════════════════════
            ## run in "stateless" mode (when no membrane time constant provided)
            z = _run_cell_stateless(j + j_td)  ## (j+j_td) >> z

        ## ═══════════════════════════════════════════════
        ## fx(z) >> zF post-activation function value(s)
        zF = self.fx(z) * self.output_scale    
        if self.use_lateral:
            zF_lat = self.fx_lat(z) * self.output_scale

        ## ════════════════════
        ## Update compartments
        self.j.set(j)
        self.j_td.set(j_td)
        self.z.set(z)
        self.zF.set(zF)
        if self.use_lateral:
            self.j_lat.set(j_lat)
            self.zF_lat.set(zF_lat)

    ## ═══════════════════════════════
    ## Update lateral synaptic weights
    @compilable
    def evolve(self):
        if self.use_lateral and self.adaptive_lateral:
            self.Wl.evolve()


    @compilable
    def reset(self):
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)

        self.j.set(restVals)
        self.j_td.set(restVals)
        self.z.set(restVals)
        self.zF.set(restVals)

        if self.use_lateral:
            self.Wl.reset()
            self.zF_lat.set(restVals)
            self.j_lat.set(restVals)


    def save(self, directory):
        super().save(directory)
        if self.use_lateral:
            self.Wl.save(directory)

    def load(self, directory):
        super().load(directory)
        if self.use_lateral:
            self.Wl.load(directory)

    def help(cls): ## component help function
        properties = {
            "cell_type": "RateCell - evolves neurons according to rate-coded/"
                         "continuous dynamics "
        }
        compartment_props = {
            "inputs":
                {"j": "External input stimulus value(s)",
                 "j_td": "External top-down input stimulus value(s); these get "
                         "multiplied by the derivative of f(x), i.e., df(x)"},
            "states":
                {"z": "Update to rate-coded continuous dynamics; value at time t"},
            "outputs":
                {"zF": "Nonlinearity/function applied to rate-coded dynamics; f(z)",
                 "zF_lat": "(if use_lateral is True) zF for lateral fx_lat(z)",
                 "j_lat": "(if use_lateral is True) lateral pressure"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "tau_m": "Cell state/membrane time constant",
            "prior": "What kind of kurtotic prior to place over neuronal dynamics?",
            "act_fx": "Elementwise activation function to apply over cell state `z`",
            "threshold": "What kind of iterative thresholding function to place over neuronal dynamics?",
            "integration_type": "Type of numerical integration to use for the cell dynamics",
            "use_lateral": "If lateral synaptic is active for inference",
            "n_patch": "Number of block-diagonal local lateral synaptic patch (n_sub_models of `Wl`)",
            "adaptive_lateral": "Adaptive lateral synapses (HebbianPatchedSynapse) or static lateral synaptic (StaticPatchedSynapse)",
            "Wl_eta": "Learning rate for the lateral synapses",
            "exc_inh": "(exc, inh) lateral synapses initial values. On-diagonal takes exc>0 and off-diagonal takes inh<0",
            "fx_lat": "Activation applied to z for the lateral pressure (defaults to act_fx)",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dz/dt = Prior(z; gamma) + (j + j_lat + j_td)",
                "hyperparameters": hyperparams}
        return info



if __name__ == '__main__':
    from ngcsimlib.context import Context
    from ngclearn import MethodProcess

    with Context("Bar") as bar:
        X = RateCell("X", 9, 20., act_fx="relu", use_lateral=True, adaptive_lateral=True,
                     exc_inh=(+5., -5.), n_patch=3, Wl_eta=0.01)
        advance = (MethodProcess(name="advance") >> X.advance_state)
        evolve = (MethodProcess(name="evolve") >> X.evolve)
