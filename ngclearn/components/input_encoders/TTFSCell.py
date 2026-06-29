import jax.numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import compilable  # from ngcsimlib.parser import compilable
from ngclearn import Compartment  # from ngcsimlib.compartment import Compartment


class TTFSCell(JaxComponent): ## time-to-first-spike (en)coding cell
    """
    A time-to-first-spike (TTFS) iterative encoder component.
    This input encoder converts a real-valued batch input vectors into sparse temporal spike trains, using
    either exponential or logarithmic latency mapping schemes. Units within this encoder group will only fire once
    within a spike train (before reset/clearing).

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | key - JAX PRNG key
    | has_fired - boolean vector denoting which units have fired thus far
    | target_spike_times - tracks targeted spike times
    | refractory_counter - refactory variable to ensure units only fire once
    | --- Cell Output Compartments: ---
    | outputs - output (binary spike train matrix output at the current time step t)
    | tols - time-of-last-spike

     Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)

        num_steps: number of total time steps of simulation to consider

        tau_ref: ttfs time constant

        latency_mode: which type of latency mapping function to apply (Default: "exponential");
            "exponential" triggers exponential-latency mapping while "logarithmic" triggers logarithmic-latency mapping
    """

    def __init__(
            self,
            name,
            n_units,
            batch_size=1,
            num_steps=100,
            tau_ref=5,
            latency_mode="exponential",
            **kwargs
    ):
        super().__init__(name, **kwargs)
        ## ttfs meta-parameters
        self.n_units = n_units
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.tau_ref = tau_ref
        self.latency_mode = latency_mode

        ## verify valid latency setting selected (only two supported!)
        if self.latency_mode not in ["exponential", "logarithmic"]:
            raise ValueError("latency_mode must be either 'exponential' or 'logarithmic'")

        ## ttfs structural compartments
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals)
        self.outputs = Compartment(restVals)
        self.has_fired = Compartment(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.bool_))
        self.target_spike_times = Compartment(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.int32))
        self.refractory_counter = Compartment(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.int32))
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms")  # time of last spike

    @staticmethod
    def _compute_latency_map(x, num_steps, latency_mode):
        ## NOTE: co-routine to map input range [0, 1] to discrete target time-step w/in [0, num_steps-1]
        x_clipped = jnp.clip(x, 1e-7, 1.0)  ## clip x, to ensure inputs do not have NaNs/Inf values
        if latency_mode == "exponential":  ## exponential-latency mapping
            ## high-intensity maps to 0 delay, low intensity maps to max delay
            raw_latency = jnp.exp(-x_clipped)
            ## normalize mapping range to scale from 0 to 1:
            ### x=1.0 -> raw=exp(-1) -> norm=0.0;  x=0.0 -> raw=exp(0)  -> norm=1.0
            norm_latency = (raw_latency - jnp.exp(-1.0)) / (jnp.exp(-1e-7) - jnp.exp(-1.0))
            ## multiply by norm_latency directly s.t. x=1.0 yields 0 delay and x=0.0 yields maximal delay
            target_times = norm_latency * (num_steps - 1)
        else:  ## logarithmic-latency mapping
            ## find minimum expected non-zero input to scale the log dynamic range properly instead of using a
            ## fixed 1e-7 clip floor, normalize based on the lowest value in x (or map the chosen clipping floor)
            raw_latency = -jnp.log(x_clipped)
            ## mapping range: x=1.0 -> raw=0.0 (Step 0); x=1e-7 -> raw=max_log (Max Step)
            #max_log = -jnp.log(1e-7)
            ## NOTE: if we want "provided" units/elements to spread across (spike train) window,
            ##       must scale relative to maximum log latency present in input
            max_input_log = jnp.maximum(-jnp.log(0.01), jnp.max(raw_latency))
            target_times = (raw_latency / max_input_log) * (num_steps - 1)
            ## ensure calculation bounds are cleanly w/in step window
            target_times = jnp.clip(target_times, 0, num_steps - 1)
        return jnp.round(target_times).astype(jnp.int32) ## output target times

    @compilable
    def advance_state(self, t, dt):
        ## NOTE: advances simulation state by resolving continuous time into a discrete step index;
        ##       computes latencies at step 0, then emits spikes based on timing targets + refractory criteria
        x = self.inputs.get()
        ## translate continuous time (t) + integration time-constant (dt) into discrete step indices
        current_step = jnp.round(t / dt).astype(jnp.int32) ## cast to int32
        ## calculate target spike schedules only on the first step of a sequence
        is_step_zero = (current_step == 0)

        computed_targets = TTFSCell._compute_latency_map(x, self.num_steps, self.latency_mode)
        init_has_fired = jnp.zeros_like(self.has_fired.get())
        init_refractory = jnp.zeros_like(self.refractory_counter.get())

        ## conditionally update targets only if resetting/restarting at step zero
        target_spike_times_curr = jnp.where(is_step_zero, computed_targets, self.target_spike_times.get())
        has_fired_curr = jnp.where(is_step_zero, init_has_fired, self.has_fired.get())
        refractory_counter_curr = jnp.where(is_step_zero, init_refractory, self.refractory_counter.get())

        refractory_counter_next = jnp.maximum(0, refractory_counter_curr - 1) ## decrement refractory counters
        ## determine units that are clear to fire (not currently in refractory window)
        is_not_refractory = (refractory_counter_next == 0)
        reached_target_time = (target_spike_times_curr == current_step) ## check which units hit target spike times
        ## spike conditions: reached time, not refractory and has not fired yet
        spikes = reached_target_time & is_not_refractory & (~has_fired_curr)
        ## update state records
        has_fired_next = jnp.where(spikes, True, has_fired_curr)
        refractory_counter_next = jnp.where(spikes, self.tau_ref, refractory_counter_next)

        ## update internal compartments
        self.outputs.set(spikes.astype(jnp.float32))
        self.target_spike_times.set(target_spike_times_curr)
        self.has_fired.set(has_fired_next)
        self.refractory_counter.set(refractory_counter_next)
        self.tols.set((1. - spikes) * self.tols.get() + (spikes * t)) ## track time-of-last-spike

    @compilable
    def reset(self): ## resets internal clock, tracking spikes, refractory states
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.tols.set(restVals)
        ## clear internal state variables
        self.target_spike_times.set(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.int32))
        self.has_fired.set(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.bool_))
        self.refractory_counter.set(jnp.zeros((self.batch_size, self.n_units), dtype=jnp.int32))

    @classmethod
    def help(cls):
        properties = {
            "cell_type": "TTFSCell - Converts static vectors into time-to-first-spike encodings"
        }
        compartment_props = {
            "inputs": {"inputs": "Takes in external real-valued input vectors to encode"},
            "outputs": {"outputs": "Emits a binary spike tensor (0 or 1) at time step t"}
        }
        hyperparams = {
            "n_units": "Number of neuronal units/features to model",
            "batch_size": "Batch size dimension of this component",
            "num_steps": "Total number of discrete time steps in the simulation window",
            "tau_ref": "Refractory period duration (in time steps) after firing a spike",
            "latency_mode": "Mathematical mapping method: 'exponential' or 'logarithmic'"
        }
        return {"properties": properties, "compartments": compartment_props, "hyperparameters": hyperparams}
