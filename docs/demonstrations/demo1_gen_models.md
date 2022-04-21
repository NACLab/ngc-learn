# Demonstration 1: Learning NGC Generative Models

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
is an instantiation of the model proposed in Rao &amp; Ballard, 1999), the GNCN-t1-Sigma  
(which is an instantiation of the model proposed in Friston 2008), and the
GNCN-PDH (which is one of the models proposed in Ororbia &amp; Kifer 2022).

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

Notice in the script, at the start of our with statement (which is used to force
the following computations to reside in a particular GPU/CPU), before initializing
a chosen model, we define a second special function to track another important
quantity special to NGC models -- the total discrepancy (ToD).
