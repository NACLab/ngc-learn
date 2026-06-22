from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable, Compartment


class LIFSRM(JaxComponent): ## LIF spike-response model (LIF-SRM)
    """
    The leaky integrate-and-fire (LIF) spike-response model (SRM); this SRM computes
    dynamics of LIF units analytically.

    | --- Cell Input Compartments: ---
    | current_j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | j_lowpass - internal low-pass-filtered current (maintained by this SRM)
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | t_last_spike - time-of-last-spike (output)
    | last_t_eval - time of last (SRM) evaluation

    | References:
    | Gerstner, W., 1995. Time structure of the activity in neural network
    | models. Physical review E, 51(1), p.738.

    | Pedagogical Reference:
    | http://www.scholarpedia.org/article/Spike-response_model

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant (ms)

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV; default: -52. mV)

        v_rest: reversal potential or membrane resting potential (in mV; default: -65 mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value;
            (default: -60 mV)
    """

    def __init__(
        self, 
        name, 
        n_units, 
        tau_m, ## membrane time constant (ms)
        thr=-52., ## threshold (mV)
        v_rest=-65., ## membrane resting potential (mV) 
        v_reset=-60., ## membrne reset potential (mV)
        batch_size=1, 
        **kwargs
    ):
        super().__init__(name, **kwargs)
        ## LIF-SRM meta-parameters
        self.n_units = n_units
        self.tau_m = tau_m ## membrane time-constant
        self.thr = thr ## threshold (mV)
        self.v_rest = v_rest ## resting potential (mV)
        self.v_reset = v_reset ## reset potential (mV)
        self.batch_size = batch_size

        ## LIF-SRM key compartments
        self.current_j = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        self.v = Compartment(jnp.full((self.batch_size, self.n_units), self.v_rest))
        self.s = Compartment(jnp.zeros((self.batch_size, self.n_units)))

        ## analytical SRM state compartments/variables (NOTE: designed to avoid maintaining explicit spike history tensors)
        self.t_last_spike = Compartment(jnp.full((self.batch_size, self.n_units), -1.0))
        self.j_lowpass = Compartment(jnp.zeros((self.batch_size, self.n_units))) ## integrated input trace
        self.last_t_eval = Compartment(jnp.zeros((self.batch_size, self.n_units))) ## tracks clock index (when evaluated)

    @compilable
    def advance_state(self, dt, t):
        ## pass last evaluation clock marker into kernel co-routine (to handle time jumps analytically)
        v_new, updated_j_trace = LIFSRM._evaluate_SRM_filter( ## apply SRM
            t, self.t_last_spike.get(), self.j_lowpass.get(), self.last_t_eval.get(),
            self.current_j.get(), self.tau_m, self.v_rest, self.v_reset, dt
        )

        s_new = (v_new > self.thr) * 1.0
        updated_t_last = jnp.where(s_new == 1.0, t, self.t_last_spike.get())
        v_output = v_new * (1.0 - s_new) + s_new * self.v_reset

        ## update compartment states
        self.v.set(v_output)
        self.s.set(s_new)
        self.j_lowpass.set(updated_j_trace)
        self.t_last_spike.set(updated_t_last)
        self.last_t_eval.set(jnp.full((self.batch_size, self.n_units), t)) ## mark this time-stamp as "evaluated"

    @compilable
    def reset(self):
        self.current_j.set(jnp.zeros((self.batch_size, self.n_units)))
        self.v.set(jnp.full((self.batch_size, self.n_units), self.v_rest))
        self.s.set(jnp.zeros((self.batch_size, self.n_units)))
        self.t_last_spike.set(jnp.full((self.batch_size, self.n_units), -1.0))
        self.j_lowpass.set(jnp.zeros((self.batch_size, self.n_units)))
        self.last_t_eval.set(jnp.zeros((self.batch_size, self.n_units)))

    @staticmethod
    def _evaluate_SRM_filter( ## kernel co-routine
        t, 
        t_last_spike, 
        j_lowpass, 
        last_t_eval, 
        current_j, 
        tau_m, 
        v_rest, 
        v_reset, 
        dt
    ):
        ## applies filter-based SRM - tracks integrated voltage contributions 
        ## calculate continuous elapsed time since this specific neuron group was last evaluated
        delta_t_eval = t - last_t_eval
        ## analytically decay historical input voltage trace over skipped time gap
        decayed_j_trace = j_lowpass * jnp.exp(-delta_t_eval / tau_m)
        ## add new incoming current pulse scaled to operate akin to single LIFCell Euler step
        new_j_trace = decayed_j_trace + (dt / tau_m) * current_j ## epsilon-kernel
        ## calc analytical spike-post-emission kernel values (self-reset mechanism)
        has_spiked = (t_last_spike >= 0.0) * 1.0
        s_post = t - t_last_spike ## kappa kernel
        eta_val = has_spiked * (v_reset - v_rest) * jnp.exp(-s_post / tau_m) ## eta kernel

        v_total = v_rest + new_j_trace + eta_val ## sum explicit kernel terms
        return v_total, new_j_trace

    @staticmethod
    def predict_next_spike( ## next-spike-time prediction co-routine
        t_start, t_last_spike, j_lowpass, tau_m, v_rest, v_reset, thr
    ):
        ## co-routine to predict future next spike, takes advantage of an LIF-SRM's 
        ## closed-form setup; specificaly, this function calculates a future (global)
        ## clock time-stamp as to when this LIF model's decaying voltage would 
        ## cross a firing threshold (note, this does not require step-wise numerical integration)
        
        ## reconstruct base self-reset kernel magnitude (eta_val)
        ### based on what historical displacement remains from last discharge event
        has_spiked = (t_last_spike >= 0.0) * 1.0
        s_post = t_start - t_last_spike
        eta_val = has_spiked * (v_reset - v_rest) * jnp.exp(-s_post / tau_m)
        ## extract combined driving force variable
        total_driving_trace = j_lowpass + eta_val

        ## define static threshold distance displacement metric
        thr_distance = thr - v_rest

        ## calc closed-form logarithmic isolation calculation for remaining segment time
        can_reach_thr = total_driving_trace > thr_distance
        ## for cases where a neuronal unit does not have enough charge to cross threshold:
        safe_ratio = jnp.where( ## handles division-by-zero / negative log errors 
            can_reach_thr, total_driving_trace / jnp.maximum(thr_distance, 1e-5), 1.0
        )
        s_remaining = tau_m * jnp.log(safe_ratio)
        ## absorb into current evaluation timestamp tracking variable
        predicted_t = t_start + s_remaining
        ## safety check: if total driving force is insufficient to cross,
        ## then flag output as "un-triggered" (i.e.,-1.0)
        return jnp.where(can_reach_thr, predicted_t, -1.0) # predicted spike time(s)

