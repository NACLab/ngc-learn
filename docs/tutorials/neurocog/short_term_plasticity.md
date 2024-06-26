# Lecture 4D: Short-Term Plasticity

In this lesson, we will study how short-term plasticity (STP) dynamics 
using one of ngc-learn's in-built synapses, the `STPDenseSynapse`. 
Specifically, we will study how a dynamic synapse may be constructed and 
examine what short-term depression (STD) and short-term facilitation
(STF) dominated configurations of an STP synapse look like. 

## Probing Short-Term Plasticity

Go ahead and make a new folder for this study and create a Python script,
i.e., `run_shortterm_plasticity.py`, to write your code for this part of the 
tutorial. 

We will write a 3-component dynamical system that connects a Poisson input 
encoding cell to a leaky integrate-and-fire (LIF) cell via a single dynamic 
synapse that evolves according to STP. We will first write our 
simulation of this dynamic synapse from the perspective of STF-dominated 
dynamics, plotting out the results under two different Poisson spike trains 
with different spiking frequencies. Then, we will modify our simulation 
to emulate dynamics from a STD-dominated perspective.

### Starting with Facilitation-Dominated Dynamics

One experiment goal with using a "dynamic synapse" is often to computationally 
model the fact that synaptic efficacy (strength/conductance magnitude) is 
not a fixed quantity (even in cases where long-term adaptation/learning is 
absent) and instead a time-varying property that depends on a fixed 
quantity of biophysical resources, e.g., neurotransmitter chemicals. This 
means, in the context of spiking cells, when a pre-synaptic neuron emits a 
pulse, this will affect the relative magnitude of the synapse's efficacy; 
in some cases, this will result in an increase (facilitation) and, in others, 
this will result in a decrease (depression) that lasts over a short period 
of time (several milliseconds in many instances). Considering the fact 
synapses have a dynamic nature to them, both over short and long time-scales, 
means that plasticity can be thought of as a stimulus and resource-dependent 
quantity, reflecting an important biophysical aspect that affects how 
neuronal systems adapt and generalize given different kinds of sensory 
stimuli.

Writing our STP dynamic synapse can be done by importing 
[STPDenseSynapse](ngclearn.components.synapses.STPDenseSynapse)  
from ngc-learn's in-built components and using it to wire the output 
spike compartment of the `PoissonCell` to the input electrical current
compartment of the `LIFCell`. This can be done as follows (using the 
meta-parameters we provide in the code block below to ensure 
STF-dominated dynamics):

```python 
from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
from ngcsimlib.compilers import compile_command, wrap_command
## import model-specific mechanisms
from ngclearn.components import PoissonCell, STPDenseSynapse, LIFCell
import ngclearn.utils.weight_distribution as dist

## create seeding keys (JAX-style)
dkey = random.PRNGKey(231)
dkey, *subkeys = random.split(dkey, 2)

firing_rate_e = 2 ## Hz (of Poisson input train)

dt = 1. ## ms # integration time constant
T_max = 5000 ## number time steps to simulate

tau_e = 5. # ms
tau_m = 20. # ms
R_m = 12. #8 .

## STF-dominated dynamics
tag = "stf_dom"
Rval = 0.15 ## resource value magnitude
tau_f = 750. # ms
tau_d = 50. # ms

plot_fname = "{}Hz_stp_{}.jpg".format(firing_rate_e, tag)

with Context("Model") as model:
    W = STPDenseSynapse("W", shape=(1, 1), weight_init=dist.constant(value=2.5),
                        resources_init=dist.constant(value=Rval),
                        tau_f=tau_f, tau_d=tau_d, key=subkeys[0])
    z0 = PoissonCell("z0", n_units=1, max_freq=firing_rate_e, key=subkeys[0])
    z1 = LIFCell("z1", n_units=1, tau_m=tau_m, resist_m=(tau_m / dt) * R_m,
                 v_rest=-60., v_reset=-70., thr=-50.,
                 tau_theta=0., theta_plus=0., refract_time=0.)

    W.inputs << z0.outputs ## z0 -> W
    z1.j << W.outputs ## W -> z1

    reset_cmd, reset_args = model.compile_by_key(z0, z1, W,
                                                 compile_key="reset")
    adv_tr_cmd, _ = model.compile_by_key(z0, W, z1, compile_key="advance_state") #z0

    model.add_command(wrap_command(jit(model.reset)), name="reset")
    model.add_command(wrap_command(jit(model.advance_state)), name="advance")

    @Context.dynamicCommand
    def clamp(obs):
        z0.inputs.set(obs)
```

Notice that the `STPDenseSynapse` has two important time constants to configure; 
`tau_f` ($\tau_f$), the facilitation time constant, and `tau_d` ($\tau_d$, the 
depression time constant. In effect, it is these two constants that you will 
want to set to obtain different desired behavior from this in-built dynamic 
synapse -- setting $\tau_f > \tau_d$ will result in STF-dominated behavior 
whereas setting $\tauf < \tau_d$ will produce STD-dominated behavior. Note 
that setting $\tau_d = 0$ will result in short-term depression being turned off 
completely ($\tau_f 0$ disables STF).

We can then write the simulated input Poisson spike train as follows:

```python 
t_vals = []
u_vals = []
x_vals = []
W_vals = []
num_z1_spikes = 0.
model.reset()
obs = jnp.asarray([[1.]])
ts = 1.
ptr = 0 # spike time pointer
for i in range(T_max):
    model.clamp(obs)
    model.advance(t=dt * ts, dt=dt)
    u = jnp.squeeze(W.u.value)
    x = jnp.squeeze(W.x.value)
    Wexc = jnp.squeeze(W.Wdyn.value)
    s0 = jnp.squeeze(W.inputs.value)
    s1 = jnp.squeeze(z1.s.value)
    num_z1_spikes = s1 + num_z1_spikes
    u_vals.append(u)
    x_vals.append(x)
    W_vals.append(Wexc)
    t_vals.append(ts)
    print("{}|  u: {}  x: {}  W: {} pre: {}  post {}".format(ts, u, x, Wexc, s0, s1))
    ts += dt
    ptr += 1
print("Number of z1 spikes = ",num_z1_spikes)

u_vals = jnp.squeeze(jnp.asarray(u_vals))
x_vals = jnp.squeeze(jnp.asarray(x_vals))
t_vals = jnp.squeeze(jnp.asarray(t_vals))
```


We may then plot out the result of the STF-dominated dynamics we 
simulate above with the following code:

```python 
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig1 = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(3, 1)

# show dynamics for one example synapse
ex_syn  = 0
ax1 = fig1.add_subplot(gs[0, 0])
_u = ax1.plot(t_vals, u_vals, '-', lw=2, color='tab:red')
#ax1.plot(range(int(round((t_max-t_0)/time_step_sim))+1),u_storage[:,ex_syn],lw=2,color='k')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('u')
ax1.set_title('STF dynamics')

ax2 = fig1.add_subplot(gs[1, 0])
_x = ax2.plot(t_vals, x_vals, '-', lw=2, color='tab:blue')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('x')
ax2.set_title('STD dynamics')

ax3 = fig1.add_subplot(gs[2, 0])
_W = ax3.plot(t_vals, W_vals, lw=2, color='tab:purple')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Syn. Weight')
ax3.set_title('Ratio of spikes transmitted: ' + str(num_z1_spikes/firing_rate_e))

fig1.subplots_adjust(top=0.3)
plt.tight_layout()
ax1.grid()
ax2.grid()
fig1.savefig(plot_fname)
```

Under the `2` Hertz Poisson spike train set up above, the plotting 
code should produce (and save to disk) the following:

<img src="../../images/tutorials/neurocog/2Hz_stp_stf_dom.jpg" width="500" />

Note that, if you change the frequency of the input Poisson spike train to `20` 
Hertz instead, like so:

```python 
firing_rate_e = 20 ## Hz (of Poisson input train)
```

and re-run your simulation script, you should obtain the following:

<img src="../../images/tutorials/neurocog/20Hz_stp_stf_dom.jpg" width="500" />

Notice that increasing the frequency in which the pre-synaptic spikes occur 
results in more volatile dynamics with respect to the effective synaptic 
efficacy over time.

### Depression-Dominated Dynamics

With your code above, it's simple to reconfigure the model to emulate 
the opposite of STF dominated dynamics, i.e., short-term depression (STD) 
dominated dynamics. 
Modify your meta-parameter values like so:

```python 
firing_rate_e = 2 ## Hz (of Poisson input train)
tag = "std_dom"
Rval = 0.45 ## resource value magnitude
tau_f = 50. # ms
tau_d = 750. # ms
```

and re-run your script to obtain an output akin to the following:

<img src="../../images/tutorials/neurocog/2Hz_stp_std_dom.jpg" width="500" />

Now, modify your meta-parameters one last time to use a higher-frequency 
input spike train, i.e., `firing_rate_e = 20 ## Hz`, to obtain a plot similar 
to the one below: 

<img src="../../images/tutorials/neurocog/20Hz_stp_std_dom.jpg" width="500" />