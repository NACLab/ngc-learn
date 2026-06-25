from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit, lax
from ngclearn import compilable, Compartment


class RAFSRM(JaxComponent): ## RAF spike-response model (RAF-SRM)
    """
    The resonate-and-fire (RAF) spike-response model (SRM); this SRM computes
    dynamics of RAF units analytically.

    | --- Cell Input Compartments: ---
    | current_j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | j_v - voltage variable state
    | j_w - angular-driver variable state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | t_last_spike - time-of-last-spike (output)
    | last_t_eval - time of last (SRM) evaluation

    | References:
    | Richardson, M.J., Brunel, N. and Hakim, V., 2003. From subthreshold to
    | firing-rate resonance. Journal of neurophysiology, 89(5), pp.2538-2554.
    |
    | Izhikevich, Eugene M. "Resonate-and-fire neurons." Neural networks 14.6-7 (2001): 883-894.

    | Pedagogical References:
    | http://www.scholarpedia.org/article/Spike-response_model      

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_v: membrane/voltage time constant (Default: 1 ms)

        tau_w: angular driver variable time constant (Default: 1 ms)

        thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes) (Default: 1 mV)

        omega: angular frequency (Default: 10)

        dampen_factor: oscillation dampening factor (Default: -1) ("b" as in RAF-Cell; Izhikevich 2001)

        v_reset: reset condition for membrane potential (Default: 1 mV)

        w_reset: reset condition for angular current driver (Default: 0)
    """

    def __init__(
        self,
        name,
        n_units,
        tau_v=1.,
        tau_w=1.,
        thr=1., ## threshold
        omega=10.,
        dampen_factor=-1.,
        v_reset=0., ## membrane reset potential
        w_reset=0., ## angular-driver variable reset potential
        batch_size=1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        ## RAF-SRM meta-parameters
        self.n_units = n_units
        self.tau_v = tau_v
        self.tau_w = tau_w
        self.omega = omega
        self.dampen_factor = dampen_factor
        self.thr = thr
        self.v_reset = v_reset
        self.w_reset = w_reset
        self.batch_size = batch_size

        ## set up SRM's key compartments
        self.current_j = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        self.v = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        self.s = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        ## set up analytical state filters (to match RAFCell construction)
        self.t_last_spike = Compartment(jnp.full((self.batch_size, self.n_units), -1.0))
        self.next_spike_t = Compartment(jnp.full((self.batch_size, self.n_units), -1.0))
        self.j_v = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        self.j_w = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        self.last_t_eval = Compartment(jnp.zeros((self.batch_size, self.n_units)))

    @compilable
    def advance_state(self, dt, t):
        v_new, updated_j_v, updated_j_w = RAFSRM._evaluate_SRM_filter( ## apply SRM to get new states
            t, self.t_last_spike.get(), self.j_v.get(), self.j_w.get(), 
            self.last_t_eval.get(), self.current_j.get(), self.tau_v, self.tau_w, 
            self.omega, self.dampen_factor, self.v_reset, self.w_reset, dt
        )
        ## calculate spike/pulse emission
        s_new = (v_new > self.thr) * 1.0
        updated_t_last = jnp.where(s_new == 1.0, t, self.t_last_spike.get())
        v_output = v_new * (1.0 - s_new) + s_new * self.v_reset
        ## update internal compartments with updated SRM states
        self.v.set(v_output)
        self.s.set(s_new)
        self.j_v.set(updated_j_v)
        self.j_w.set(updated_j_w)
        self.t_last_spike.set(updated_t_last)
        self.last_t_eval.set(jnp.full((self.batch_size, self.n_units), t))

    @compilable
    def reset(self): #
        self.current_j.set(jnp.zeros((self.batch_size, self.n_units)))
        self.v.set(jnp.zeros((self.batch_size, self.n_units)))
        self.s.set(jnp.zeros((self.batch_size, self.n_units)))
        self.t_last_spike.set(jnp.full((self.batch_size, self.n_units), -1.0))
        self.next_spike_t.set(jnp.full((self.batch_size, self.n_units), -1.0))
        self.j_v.set(jnp.zeros((self.batch_size, self.n_units)))
        self.j_w.set(jnp.zeros((self.batch_size, self.n_units)))
        self.last_t_eval.set(jnp.zeros((self.batch_size, self.n_units)))

    @compilable
    def predict_next_spike(self, t_start):
        next_spike_t = RAFSRM._predict_next_spike(  ## next-spike-time predictor
            t_start,
            self.j_v.get(),
            self.j_w.get(),
            self.tau_v,
            self.tau_w,
            self.omega,
            self.dampen_factor,
            self.thr,
            max_iters=10
        )
        self.next_spike_t.set(next_spike_t)  ## store predicted next spike time(s)

    @staticmethod
    def _evaluate_SRM_filter( ## kernel co-routine
        t, 
        t_last_spike, 
        j_v, 
        j_w, 
        last_t_eval, 
        current_j, 
        tau_v, 
        tau_w, 
        omega, 
        dampen_factor, 
        v_reset, 
        w_reset, 
        dt
    ):
        ## applies filter-based cumulative SRM for resonate-and-fire (RAF) units
        ## NOTE: this analytical SRM is designed to match exact system matrix of RAFCell
        delta_t_eval = t - last_t_eval ## calc (Dirac) delta time since last spike evaluation
        ## reconstruct continuous system (of matrix parameters) of RAF dampened oscilator
        ### dv/dt = a*v + b*w, and, dw/dt = c*v + d*w + j/tau_w
        a = dampen_factor / tau_v
        b = omega / tau_v
        c = -omega / tau_w
        d = dampen_factor / tau_w
        ## calculate trace dampening (gamma) & determinant natural frequency
        gamma = -0.5 * (a + d)
        omega_0_sq = (a * d) - (b * c)
        omega_d = jnp.sqrt(jnp.maximum(omega_0_sq - gamma**2, 1e-6))
        ## analytically propagate ongoing sub-threshold state across time jump
        decay = jnp.exp(-gamma * delta_t_eval)
        cos_wd = jnp.cos(omega_d * delta_t_eval)
        sin_wd = jnp.sin(omega_d * delta_t_eval)

        v_old = j_v ## store current v
        w_old = j_w ## store current w
        ## calculate decoupled matrix exponential transformation equations
        v_prop = decay * (
            v_old * (cos_wd + ((a + gamma) / omega_d) * sin_wd) + w_old * (b / omega_d) * sin_wd
        )
        w_prop = decay * (
            v_old * (c / omega_d) * sin_wd + w_old * (cos_wd + ((d + gamma) / omega_d) * sin_wd)
        )
        ## apply input current as "Euler-step" velocity shift to match exact step time
        new_v = v_prop
        new_w = w_prop + (current_j / tau_w) * dt
        ## calcuate spike post-emission / self-reset kernel contribution
        has_spiked = (t_last_spike >= 0.0) * 1.0
        s_post = t - t_last_spike ## kappa-kernel
        eta_decay = jnp.exp(-gamma * s_post)
        eta_cos = jnp.cos(omega_d * s_post)
        eta_sin = jnp.sin(omega_d * s_post)
        ## set up eta kernel
        eta_v = has_spiked * (
            v_reset * (eta_cos + ((a + gamma) / omega_d) * eta_sin) + w_reset * (b / omega_d) * eta_sin
        )
        ## compute total combined membrane voltage output
        v_total = new_v + eta_v
        return v_total, new_v, new_w

    @staticmethod
    def _predict_next_spike( ## co-routine used to compute when a future spike will occur
        t_start,
        j_v,
        j_w,
        tau_v,
        tau_w,
        omega,
        dampen_factor,
        thr,
        max_iters=10
    ):
        ## NOTE: this co-routine is based on a Newton-Raphson root finding process to predict when exactly
        ## continuous sub-threshold wave equation will cross threshold line

        ## standardize SRM system parameters
        a = dampen_factor / tau_v
        b = omega / tau_v
        c = -omega / tau_w
        d = dampen_factor / tau_w

        gamma = -0.5 * (a + d)
        omega_0_sq = (a * d) - (b * c)
        omega_d = jnp.sqrt(jnp.maximum(omega_0_sq - gamma**2, 1e-6))

        ## define analytical function V(s) and its temporal derivative dV(s)/ds
        def evaluate_v_and_dv(s):
            decay = jnp.exp(-gamma * s)
            cos_wd = jnp.cos(omega_d * s)
            sin_wd = jnp.sin(omega_d * s)
            ## calc continuous voltage coordinate position equation
            v_prop = decay * (j_v * (cos_wd + ((a + gamma) / omega_d) * sin_wd) + j_w * (b / omega_d) * sin_wd)
            ## calc continuous velocity coordinate position equation (dV/ds)
            ## based on underlying coupled system matrix row: dv/dt = a*v + b*w
            w_prop = decay * (j_v * (c / omega_d) * sin_wd + j_w * (cos_wd + ((d + gamma) / omega_d) * sin_wd))
            dv_ds = a * v_prop + b * w_prop
            return v_prop, dv_ds

        ## initialize Newton-Raphson search sequence loops; initial guess: s ~ 0.1ms
        s_guess = jnp.full_like(j_v, 0.1)

        def scan_body(carry, _): ## set up scanner for this search process
            s_curr = carry
            v_curr, dv_curr = evaluate_v_and_dv(s_curr)
            f_s = v_curr - thr
            ## update root estimation step while avoiding divide-by-zero bounds
            s_next = s_curr - f_s / jnp.where(jnp.abs(dv_curr) < 1e-5, 1e-5, dv_curr)
            ## keep estimated elapsed intervals bounded strictly positive
            s_next = jnp.maximum(s_next, 0.0)
            return s_next, None

        ## execute unrolled iteration passes completely w/in JAX/JIT compiler context
        final_s, _ = lax.scan(scan_body, s_guess, None, length=max_iters)
        predicted_t = t_start + final_s ## compute absolute (executed) global clock projection coordinate
        ## employ a safety check:
        ### if initial slope is negative (or diverging), flag it as un-triggered (i.e., -1.0)
        v_final, _ = evaluate_v_and_dv(final_s)
        is_valid = (final_s > 0.0) & (jnp.abs(v_final - thr) < 1e-2)
        return jnp.where(is_valid, predicted_t, -1.0) ## spit out guessed future spike time

