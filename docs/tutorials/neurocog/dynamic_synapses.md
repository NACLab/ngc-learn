# Dynamic Synapses and Conductance

In this lesson, we will study dynamic synapses, or synaptic cable components in 
ngc-learn that evolve on fast time-scales in response to their pre-synaptic inputs. 
These types of chemical synapse components are useful for modeling time-varying 
conductance which ultimately drives eletrical current input into neuronal units 
(such as spiking cells).  
Here, we will learn how to build two important types of dynamic synapses in 
ngc-learn -- the exponential synapse and the alpha synapse -- and visualize 
the time-course of their resulting conductances. In addition, we will then 
construct and study a small neuronal circuit involving a leaky integrator that 
is driven by exponential synapses relaying pulses from an excitatory and an 
inhibitory population of Poisson input encoding cells.

## Chemical Synapses


Building a dynamic synapse can be done by importing 
[ExponentialSynapse](ngclearn.components.synapses.ExponentialSynapse) and 
[AlphaSynapse](ngclearn.components.synapses.AlphaSynapse)
from ngc-learn's in-built components and setting them up within a model 
context for easy analysis.
This can be done as follows (using the meta-parameters we provide in the 
code block below to ensure reasonable dynamics):

```python
from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import ExponentialSynapse, AlphaSynapse
from ngclearn.operations import summation

from ngcsimlib.compilers.process import Process
from ngcsimlib.context import Context
import ngclearn.utils.weight_distribution as dist


dkey = random.PRNGKey(1234) ## creating seeding keys for synapses
dkey, *subkeys = random.split(dkey, 6)
dt = 0.1 # ms ## integration time constant
T = 8. # ms ## total duration time

Tsteps = int(T/dt) + 1

# ---- build a two-synapse system ----
with Context("dual_syn_system") as ctx:
    Wexp = ExponentialSynapse( ## exponential dynamic synapse
        name="Wexp", shape=(1, 1), tau_syn=3., g_syn_bar=1., syn_rest=0., resist_scale=1.,
        weight_init=dist.constant(value=1.), key=subkeys[0]
    )
    Walpha = AlphaSynapse( ## alpha dynamic synapse
        name="Walpha", shape=(1, 1), tau_syn=1., g_syn_bar=1., syn_rest=0., resist_scale=1.,
        weight_init=dist.constant(value=1.), key=subkeys[0]
    )

    ## set up basic simulation process calls
    advance_process = (Process("advance_proc")
                       >> Wexp.advance_state
                       >> Walpha.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

    reset_process = (Process("reset_proc")
                     >> Wexp.reset
                     >> Walpha.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
```

where we notice in the above we have instantiated two different kinds of chemical synapse components 
that we will run side-by-side in order to extract their produced conductance values in response to 
the exact same input stream. For both the exponential and the alpha synapse, there are at least three 
important hyper-parameters to configure:
1. `tau_syn` ($\tau_{\text{syn}}$): the synaptic conductance decay time constant; 
2. `g_syn_bar` ($\bar{g}_{\text{syn}}$): the maximal conductance value produced by each pulse transmitted 
   across this synapse; and,  
3. `syn_rest` ($E_{rest}$): the (post-synaptic) reversal potential for this synapse -- note that this value 
    determines the direction of current flow through the synapse, yielding a synapse with an 
    excitatory nature for non-negative values of `syn_rest` or a synapse with an inhibitory 
    nature for negative values of `syn_rest`.


The flow of electrical current from a pre-synaptic neuron to a post-synaptic one is often modeled under the assumption that pre-synaptic pulses result in impermanent (transient; lasting for a short period of time) changes in the conductance of a post-synaptic neuron. As a result, the resulting conductance dynamics $g_{\text{syn}}(t)$ of each of the two synapses that you have built above can be simulated in ngc-learn according to one or more ordinary differential equations (ODEs). 
For the exponential synapse, the dynamics adhere to the following ODE: 

$$
\frac{\partial g_{\text{syn}}(t)}{\partial t} = -g_{\text{syn}}(t)/\tau_{\text{syn}} + \bar{g}_{\text{syn}} \sum_{k} \delta(t - t_{k}) 
$$

where the conductance (for a post-synaptic unit) output of this synapse is driven by a sum over all of its incoming pre-synaptic spikes; this ODE means that pre-synaptic spikes are filtered via an expoential kernel (i.e., a low-pass filter). 
On the other hand, for the alpha synapse, the dynamics adhere to the following coupled set of ODEs:

$$
\frac{\partial h_{\text{syn}}(t)}{\partial t} &= -h_{\text{syn}}(t)/\tau_{\text{syn}} + \bar{g}_{\text{syn}} \sum_{k} \delta(t - t_{k}) \\
\frac{\partial g_{\text{syn}}(t)}{\partial t} &= -g_{\text{syn}}(t)/\tau_{\text{syn}} + h_{\text{syn}}(t)/\tau_{\text{syn}}
$$

where $h_{\text{syn}}(t)$ is an intermediate variable that operates in service of driving the conductance variable $g_{\text{syn}}(t)$ itself.

For both the exponential and the alpha synapse, the changes in conductance are finally converted (via Ohm's law) to electrical current to produce the final derived variable $j_{\text{syn}}(t)$:

$$
j_{\text{syn}}(t) = g_{\text{syn}}(t) (v(t) - E_{\text{rest}})
$$

where $v_{\text{rest}$ (or $E_{\text{rest}}$) is the post-synaptic reverse potential of the synapse; this is typically set to $E_{\text{rest}} = 0$ (millivolts; mV)for the case of excitatory changes and $E_{\text{rest}} = -75$ (mV) for the case of inhibitory changes. $v(t)$ is the voltage/membrane potential of the post-synaptic the synaptic cable wires to, meaning that the conductance models above are voltage-dependent (in ngc-learn, if one wants voltage-independent conductance, then `syn_rest` must be set to `None`). 


### Examining the Conductances of Dynamic Synapses

We can track and visualize the conductance outputs of our two different dynamic synapses by running a stream of controlled pre-synaptic pulses. Specifically, we will observe the output behavior of each in response to a sparse stream, eight milliseconds in length, where only a single spike is emitted at one millisecond. 
To create the simulation of a single input pulse stream, you can write the following code:

```python
time_ticks = []
time_labs = []
for t in range(Tsteps):
    if t % 10 == 0:
        time_ticks.append(t)
        time_labs.append(f"{t * dt:.1f}")

time_span = []
g = []
ga = []
ctx.reset()
for t in range(Tsteps):
    s_t = jnp.zeros((1, 1))
    if t * dt == 1.: ## pulse at 1 ms
        s_t = jnp.ones((1, 1))
    Wexp.inputs.set(s_t)
    Walpha.inputs.set(s_t)
    Wexp.v.set(Wexp.v.value * 0)
    Walpha.v.set(Walpha.v.value * 0)
    ctx.run(t=t * dt, dt=dt)

    print(f"\r g = {Wexp.g_syn.value}  ga = {Walpha.g_syn.value}", end="")
    g.append(Wexp.g_syn.value)
    ga.append(Walpha.g_syn.value)
    time_span.append(t) #* dt)
print()
g = jnp.squeeze(jnp.concatenate(g, axis=1))
g = g/jnp.amax(g)
ga = jnp.squeeze(jnp.concatenate(ga, axis=1))
ga = ga/jnp.amax(ga)
```

Note that we further normalize the conductance trajectories of both synapses to lie within 
the range of $[0, 1]$, primarily for visualization purposes. 
Finally, to visualize the conductance time-course of both synapses, you can write the 
following: 

```python 
import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

## ---- plot the exponential synapse conductance time-course ----
fig, ax = plt.subplots()

gvals = ax.plot(time_span, g, '-', color='tab:red')
#plt.xticks(time_span, time_labs)
ax.set_xticks(time_ticks, time_labs)
ax.set(xlabel='Time (ms)', ylabel='Conductance',
      title='Exponential Synapse Conductance Time-Course')
ax.grid(which="major")
fig.savefig("exp_syn.jpg")
plt.close()

## ---- plot the alpha synapse conductance time-course ----
fig, ax = plt.subplots()

gvals = ax.plot(time_span, ga, '-', color='tab:blue')
#plt.xticks(time_span, time_labs)
ax.set_xticks(time_ticks, time_labs)
ax.set(xlabel='Time (ms)', ylabel='Conductance',
      title='Alpha Synapse Conductance Time-Course')
ax.grid(which="major")
fig.savefig("alpha_syn.jpg")
plt.close()
```

which should produce and save two plots to disk. You can then compare and contrast the plots of the 
expoential and alpha synapse conductance trajectories:

```{eval-rst}
.. table::
   :align: center

   +---------------------------------------------------------+-----------------------------------------------------------+
   | .. image:: ../docs/images/tutorials/neurocog/expsyn.png | .. image:: ../docs/images/tutorials/neurocog/alphasyn.png |
   |   :width: 100px                                         |   :width: 100px                                           |
   |   :align: center                                        |   :align: center                                          |
   +---------------------------------------------------------+-----------------------------------------------------------+
```

## Excitatory-Inhibitory Driven Dynamics



