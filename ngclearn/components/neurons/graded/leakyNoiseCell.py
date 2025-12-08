from jax import numpy as jnp, random, jit
from ngcsimlib.logger import info
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, step_euler, step_rk2, step_rk4

def _dfz_fn(z, j_input, j_recurrent, eps, tau_x, sigma_rec, leak_scale): ## raw dynamics ODE
    dz_dt = -(z * leak_scale) + (j_recurrent + j_input) + jnp.sqrt(2. * tau_x * jnp.square(sigma_rec)) * eps
    return dz_dt * (1. / tau_x)

def _dfz(t, z, params): ## raw dynamics ODE wrapper
    j_input, j_recurrent, eps, tau_x, sigma_rec, leak_scale = params
    return _dfz_fn(z, j_input, j_recurrent, eps, tau_x, sigma_rec, leak_scale)

class LeakyNoiseCell(JaxComponent): ## Real-valued, leaky noise cell
    """
    A non-spiking cell driven by the gradient dynamics entailed by a continuous-time noisy, leaky recurrent state.

    Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4771709/

    The specific differential equation that characterizes this cell is (for adjusting x) is:

    | tau_x * dx/dt = -x + j_rec + j_in + sqrt(2 alpha (sigma_pre)^2) * eps; and,
    | r = f(x) + (eps * sigma_post). 
    | where j_in is the set of incoming input signals
    | and j_rec is the set of recurrent input signals
    | and eps is a sample of unit Gaussian noise, i.e., eps ~ N(0, 1)
    | and f(x) is the rectification function
    | and sigma_pre is the pre-rectification noise applied to membrane x
    | and sigma_post is the post-rectification noise applied to rates f(x)

    | --- Cell Input Compartments: ---
    | j_input - input (bottom-up) electrical/stimulus current (takes in external signals)
    | j_recurrent - recurrent electrical/stimulus pressure
    | --- Cell State Compartments ---
    | x - noisy rate activity / current value of state
    | --- Cell Output Compartments: ---
    | r - post-rectified activity, e.g., fx(x) = relu(x)
    | r_prime - post-rectified temporal derivative, e.g., dfx(x) = d_relu(x)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_x: state membrane time constant (milliseconds)

        act_fx: rectification function (Default: "relu")

        output_scale: factor to multiply output of nonlinearity of this cell by (Default: 1.)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration) and "midpoint" or "rk2"
            (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will increase the accuracy of the estimate of
                the cell's evolution at an increase in computational cost (and simulation time)

        sigma_pre: pre-rectification noise scaling factor / standard deviation (Default: 0.1)

        sigma_post: post-rectification noise scaling factor / standard deviation (Default: 0.)

        leak_scale: degree to which membrane leak should be scaled (Default: 1)
    """

    def __init__(
            self, name, n_units, tau_x, act_fx="relu", integration_type="euler", batch_size=1, sigma_pre=0.1,
            sigma_post=0.1, leak_scale=1., shape=None, **kwargs
    ):
        super().__init__(name, **kwargs)


        self.tau_x = tau_x
        self.sigma_pre = sigma_pre ## a pre-rectification scaling factor
        self.sigma_post = sigma_post ## a post-rectification scaling factor
        self.leak_scale = leak_scale ## the leak scaling factor (most appropriate default is 1)

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
        self.batch_size = batch_size

        self.fx, self.dfx = create_function(fun_name=act_fx)

        # compartments (state of the cell & parameters will be updated through stateless calls)
        restVals = jnp.zeros(_shape)
        self.j_input = Compartment(restVals, display_name="Input Stimulus Current", units="mA") # electrical current
        self.j_recurrent = Compartment(restVals, display_name="Recurrent Stimulus Current", units="mA") # electrical current
        self.x = Compartment(restVals, display_name="Rate Activity", units="mA") # rate activity
        self.r = Compartment(restVals, display_name="(Rectified) Rate Activity") # rectified output
        self.r_prime = Compartment(restVals, display_name="Derivative of rate activity")

    @compilable
    def advance_state(self, t, dt):
        ## run a step of integration over neuronal dynamics
        ### Note: self.fx is the "rectifier" (rectification function)
        key, skey = random.split(self.key.get(), 2)
        eps_pre = random.normal(skey, shape=self.x.get().shape) ## pre-rectifier distributional noise
        key, skey = random.split(self.key.get(), 2)
        eps_post = random.normal(skey, shape=self.x.get().shape)  ## post-rectifier distributional noise

        #x = _run_cell(dt, self.j_input.get(), self.j_recurrent.get(), self.x.get(), eps, self.tau_x, self.sigma_rec, integType=self.intgFlag)
        _step_fns = {
            0: step_euler,
            1: step_rk2,
            2: step_rk4,
        }
        _step_fn = _step_fns[self.intgFlag] #_step_fns.get(self.intgFlag, step_euler)
        params = (self.j_input.get(), self.j_recurrent.get(), eps_pre, self.tau_x, self.sigma_pre, self.leak_scale)
        _, x = _step_fn(0., self.x.get(), _dfz, dt, params) ## update state activation dynamics
        r = self.fx(x) + (eps_post * self.sigma_post)  ## calculate (rectified) activity rates; f(x)
        r_prime = self.dfx(x) ## calculate local deriv of activity rates; f'(x)

        ## set compartments to next state values in accordance with dynamics
        self.key.set(key) ## carry noise key over transition (to next state of component)
        self.x.set(x)
        self.r.set(r)
        self.r_prime.set(r_prime)

    @compilable
    def reset(self):
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)
        self.j_input.set(restVals)
        self.j_recurrent.set(restVals)
        self.x.set(restVals)
        self.r.set(restVals)
        self.r_prime.set(restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "LeakyNoiseCell - evolves neurons according to continuous-time noisy/leaky dynamics "
        }
        compartment_props = {
            "inputs":
                {"j_input": "External input stimulus value(s)",
                 "j_recurrent": "Recurrent/prior-state stimulus value(s)"},
            "states":
                {"x": "Update to continuous noisy, leaky dynamics; value at time t"},
            "outputs":
                {"r": "A linear rectifier applied to rate-coded dynamics; f(z)"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "tau_x": "State time constant",
            "sigma_pre": "The non-zero degree/scale of (pre-rectification) noise to inject into this neuron"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_x * dz/dt = -z + j_input + j_recurrent + noise, where noise ~N(0, sigma_rec)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = LeakyNoiseCell("X", 9, 0.03)
    print(X)
