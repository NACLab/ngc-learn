# Lecture 2E: The Hodgkin-Huxley Cell

In this tutorial, we will study/setup one of the most important biophysical 
neuronal models in computational neuroscience -- the Hodgkin-Huxley (H-H) spiking 
cell model.

## Using and Probing the H-H Cell

Go ahead and make a new folder for this study and create a Python script,
i.e., `run_hhcell.py`, to write your code for this part of the tutorial.

Now let's set up the controller for this lesson's simulation and construct a
single component system made up of an H-H cell.


### Instantiating the H-H Neuronal Cell

Constructing/setting up a dynamical system made up of only a single
H-H cell amounts to the following:

```python
from jax import numpy as jnp, random, jit
import numpy as np

from ngclearn.utils.model_utils import scanner
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
## import model-specific mechanisms
from ngclearn.components.neurons.spiking.hodgkinHuxleyCell import HodgkinHuxleyCell

## create seeding keys (JAX-style)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

T = 20000  ## number of simulation steps to run
dt = 0.01  # ms ## compute integration time constant

## H-H cell hyperparameters
v_Na=115. ## sodium reversal potential
v_K=-35. ## potassium reversal potential
v_L=10.6 ## leak reversal potential
g_Na=100. ## sodium conductance (per unit area)
g_K=5. ## potassium conductance (per unit area)
g_L=0.3 ## leak conductance (per unit area)
v_thr=4. ## membrane potential threshold for binary pulse production

## create simple system with only one AdEx
with Context("Model") as model:
    cell = HodgkinHuxleyCell(
        name="z0", n_units=1, tau_v=1., resist_m=1., v_Na=v_Na, v_K=v_K, v_L=v_L, 
        g_Na=g_Na, g_K=g_K, g_L=g_L, thr=v_thr, integration_type="euler", key=subkeys[0]
    )

    ## create and compile core simulation commands
    advance_process = (JaxProcess()
                       >> cell.advance_state)
    model.wrap_and_add_command(jit(advance_process.pure), name="advance")

    reset_process = (JaxProcess()
                     >> cell.reset)
    model.wrap_and_add_command(jit(reset_process.pure), name="reset")

    ## set up non-compiled utility commands
    @Context.dynamicCommand
    def clamp(x):
        cell.j.set(x)
```

Notably, the H-H model is a four-dimensional differential equation system, invented in 1952 
by Alan Hodgkin and Andrew Huxley to describe the ionic mechanisms that underwrite the 
initiation and propagation of action potentials within squid giant axons (the axons responsible for 
controlling the squid's water jet propulsion system in squid) <b>[1]</b>. 
Notably, the H-H cell is one of the more complex cells supported by ngc-learn since it  
models membrane potential `v` ($\mathbf{v}_t$) as well as three gates or channels. The three channels are 
essentially probability values: 
`n` ($\mathbf{n}_t$)  for the probability of potassium channel subunit activation, 
`m` ($\mathbf{m}_t$)  for the probability of sodium channel subunit activation, and 
`h` ($\mathbf{h}_t$)  for the probability of sodium channel subunit inactivation.  

neurons and muscle cells. It is a continuous-time dynamical system.

Formally, the core dynamics of the H-H cell can be written out as follows:

$$
\tau_v \frac{\partial \mathbf{v}_t}{\partial t} &= \mathbf{j}_t - g_Na * \mathbf{m}^3_t * \mathbf{h}_t * (\mathbf{v}_t - v_Na) - g_K * \mathbf{n}^4_t * (\mathbf{v}_t - v_K) - g_L * (\mathbf{v}_t - v_L) \\
\frac{\partial \mathbf{n}_t}{\partial t} &= alpha_n(\mathbf{v}_t) * (1 - \mathbf{n}_t) - beta_n(\mathbf{v}_t) * \mathbf{n}_t \\
\frac{\partial \mathbf{m}_t}{\partial t} &= alpha_m(\mathbf{v}_t) * (1 - \mathbf{m}_t) - beta_m(\mathbf{v}_t) * \mathbf{m}_t \\
\frac{\partial \mathbf{h}_t}{\partial t} &= alpha_h(\mathbf{v}_t) * (1 - \mathbf{h}_t) - beta_h(\mathbf{v}_t) * \mathbf{h}_t
$$

where we observe that the above four-dimensional set of dynamics is composed of nonlinear ODEs. Notice that, in each gate or channel probability ODE, there are two generator functions (each of which is a function of the membrane potential $\mathbf{v}_t$) that produces the necessary dynamic coefficients at time $t$; $\alpha_x(\mathbf{v}_t)$ and $\beta_x(\mathbf{v}_t)$ produce different biopphysical weighting values depending on which channel $x = \{n, m, h\}$ they are related to. 
Note that, in ngc-learn's implementation of the H-H cell model, most of the core coefficients have been generally set according to Hodgkin and Huxley's 1952 work but can be configured by the experimenter to obtain different kinds of behavior/dynamics.

### Simulating the H-H Neuronal Cell

To see how the H-H cell works, we next write some code for visualizing how
the node's membrane potential and core related gates/channels evolve with time
(over a period of about `200` milliseconds). We will inject a square input pulse current 
into our H-H cell (specifically into its `j` compartment) and observe how the cell behaves in response. 
Specifically, we simulate the injection of this kind of current via the code below:

```python
## create input stimulation feed
stim = np.zeros((1, T))
stim[0, 7000:13000] = 50 ## inject square input pulse
x_seq = jnp.array(stim)

## now simulate the model
time_span = []
outs = []
v = []
n = []
m = []
h = []
model.reset()
for ts in range(x_seq.shape[1]):
    x_t = jnp.array([[x_seq[0, ts]]]) ## get data at time t
    model.clamp(x_t)
    model.run(t=ts * dt, dt=dt)
    outs.append(a.s.value)
    n.append(cell.n.value[0, 0])
    m.append(cell.m.value[0, 0])
    h.append(cell.h.value[0, 0])
    v.append(cell.v.value[0, 0])
    print(f"\r {ts} v = {cell.v.value}", end="")
    time_span.append(ts*dt)
outs = jnp.concatenate(outs, axis=1)
v = jnp.array(v)
time_span = jnp.array(time_span)
outs = jnp.squeeze(outs)
```

and we can plot the dynamics of the  neuron's voltage `v` and its three gate/channel 
variables, `h`, `m`, and `n`, with the following:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
## create subplot 1 -- show membrane voltage potential
ax1 = plt.subplot(311)
ax1.plot(time_span, v - 70., color='b')
ax1.set_ylabel("Potential (mV)")
ax1.set_title("Hodgkin-Huxley Spike Model") 

## create subplot 2 -- show electrical input current
ax2 = plt.subplot(312)
ax2.plot(time_span, jnp.squeeze(x_seq), color='r')
ax2.set_ylabel("Stimulation (µA/cm^2)")

## create subplot 3 -- show activation of gate/channel variables
ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(time_span, h, label='h')
ax3.plot(time_span, m, label='m')
ax3.plot(time_span, n, label='n')
ax3.set_ylabel("Activation (frac)")
ax3.legend()

plt.tight_layout()
plt.savefig("{0}".format("hh_plot.jpg"))
plt.close()
```

You should get a compound plot that depict the evolution of the H-H cell's voltage
and channel/gate variables, i.e., saved as `hh_plot.jpg` locally to
disk, like the one below:

```{eval-rst}
.. table::
   :align: center

   +--------------------------------------------------------+
   | .. image:: ../../images/tutorials/neurocog/hh_plot.jpg |
   |   :scale: 75%                                          |
   |   :align: center                                       |
   +--------------------------------------------------------+
```

A useful note is that the H-H cell above used Euler integration to step through its
dynamics (this is the default/base routine for all cell components in ngc-learn). 
However, one could configure the cell to use the midpoint method for integration
by setting its argument `integration_type = rk2` or the Runge-Kutta fourth-order 
routine via `integration_type=rk4` for cases where, at the cost of increased 
compute time, more accurate dynamics are possible.

## Optional: Setting Up The Components with a JSON Configuration

While you are not required to create a JSON configuration file for ngc-learn,
to get rid of the warning that ngc-learn will throw at the start of your
program's execution (indicating that you do not have a configuration set up yet),
all you need to do is create a sub-directory for your JSON configuration
inside of your project code's directory, i.e., `json_files/modules.json`.
Inside the JSON file, you would write the following:

```json
[
    {"absolute_path": "ngclearn.components",
        "attributes": [
            {"name": "HodgkinHuxleyCell"}]
    },
    {"absolute_path": "ngcsimlib.operations",
        "attributes": [
            {"name": "overwrite"}]
    }
]
```

## References

<b>[1]</b> Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description 
of membrane current and its application to conduction and excitation in nerve." 
The Journal of physiology 117.4 (1952): 500.

