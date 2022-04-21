# Demo 1: Learning NGC Generative Models

In this demonstration, we will learn how to use ngc-learn's Model Museum to fit an
NGC generative model, specifically called a generative neural coding network (GNCN),
to the MNIST dataset. Specifically, we will focus on training three key models,
each with different structural properties, and estimating their marginal log likelihoods.

To start, navigate to the `examples/` directory to access the example/demonstration
code and further enter the `examples/data/` sub-folder. Unzip the file
`mnist.zip` to create one more sub-folder that contains a set of numpy arrays that house
a different slice of the MNIST database, i.e., `trainX.npy` and `trainY.npy` compose
the training set (image patterns and their labels), `validX.npy`and `validY.npy` make
up the development/validation set, and `testX.npy`and `testY.npy` compose the test set.
Note that pixels in all image vectors have been normalized to the range of [0,1].

Next,in `examples/`, we observe the provided script `sim_train.py`, which contains the
code to execute the training process of an NGC model. Inside this file, we can export
one of three possible GNCNs from ngc-learn's Model Museum, i.e., the GNCN-t1 (which
is an instantiation of the model proposed in Rao &amp; Ballard, 1999 [1]),
the GNCN-t1-Sigma (which is an instantiation of the model proposed in Friston
2008 [2]), and the GNCN-PDH (which is one of the models proposed in
Ororbia &amp; Kifer 2022 [3]).

Importing models from the Model Museum is straightforward and only
requires a few lines to be placed in the header of a training script. Notice
that we import several items besides the models, including a DataLoader, like so:

```python
from ngclearn.utils.data_utils import DataLoader
```

some metrics, transformations, and other I/O tools, as follows:

```python
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
```

where Config is an argument configuration object that reads in values
set by a user in a configuration file, transform_utils contains mathematical
functions to alter vectors/matrices (we will use the `binarize()` function), and
metric_utils contains measurement/performance functions (we will use the binary
cross entropy routine `bce()`). Finally, we import the models themselves,
as shown below:

```python
from ngclearn.museum.gncn_t1 import GNCN_t1
from ngclearn.museum.gncn_t1_sigma import GNCN_t1_Sigma
from ngclearn.museum.gncn_pdh import GNCN_PDH
```

With the above imported from ngc-learn, we have everything we need to craft a
full training cycle as well as track the model's out-of-sample inference ability
on validation data.

Notice in the script, at the start of our with-statement (which is used to force
the following computations to reside in a particular GPU/CPU), before initializing
a chosen model, we define a second special function to track another important
quantity special to NGC models -- the total discrepancy (ToD) -- as follows:

```python
def calc_ToD(agent):
    """Measures the total discrepancy (ToD) of a given NGC model"""
    ToD = 0.0
    L2 = agent.ngc_model.extract(node_name="e2", node_var_name="L")
    L1 = agent.ngc_model.extract(node_name="e1", node_var_name="L")
    L0 = agent.ngc_model.extract(node_name="e0", node_var_name="L")
    ToD = -(L0 + L1 + L2)
    return ToD
```

This function is used to measure the internal disorder, of approximate free energy,
within an NGC model based on its error neurons (if using specialized error neuron
nodes, notice that a specialized node contains a scalar loss value, otherwise,
you will need to take convert the error neuron vector to a scalar via something
like a distance function). ToD allows us to track if entire NGC system's optimization
process is behaving correctly and making progress towards reaching a stable fixed-point.

Next, we write an evaluation function that leverages a DataLoader and a NGC model
and returns some useful problem-specific measurements, i.e., in this demo's case,
we want to measure and track binary cross entropy across training iterations/epochs.
The evaluation loop can be written like so:

```python
def eval_model(agent, dataset, calc_ToD, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    ToD = 0.0 # total disrepancy over entire data pool
    Lx = 0.0 # metric/loss over entire data pool
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        N += x.shape[0]
        x_hat = agent.settle(x) # conduct iterative inference
        ToD_t = calc_ToD(agent) # calc ToD
        # update tracked fixed-point losses
        Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx
        ToD = calc_ToD(agent) + ToD
        agent.clear()
        if verbose == True:
            print("\r ToD {0}  Lx {1} over {2} samples...".format((ToD/(N * 1.0)), (Lx/(N * 1.0)), N),end="")
    if verbose == True:
        print()
    Lx = Lx / N
    ToD = ToD / N
    return ToD, Lx
```

Notice that we pass in the current NGC model (`agent`), the DataLoader (`dataset`), and the ToD
function we wrote earlier. Now we have a means to measure some aspect of the generalization
ability of our NGC model so all that remains is to craft a training process loop
and we can simulate an NGC. This loop can be as follows:



**References:**
[1] Rao, Rajesh PN, and Dana H. Ballard. "Predictive coding in the visual cortex:
a functional interpretation of some extra-classical receptive-field effects."
Nature neuroscience 2.1 (1999): 79-87. <br>
[2] Friston, Karl. "Hierarchical models in the brain." PLoS Computational
Biology 4.11 (2008): e1000211. <br>
[3] Ororbia, A., and Kifer, D. The neural coding framework for learning
generative models. Nature Communications 13, 2064 (2022).
