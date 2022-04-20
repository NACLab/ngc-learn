# Demonstration 1: Learning NGC Generative Models

In this demonstration, we will learn how to use NGC-Learn's Model Museum to fit an
NGC generative model, specifically called a generative neural coding network (GNCN),
to the MNIST dataset. Specifically, we will focus on training three key models,
each with different structural properties, and estimating their marginal log likelihoods.

We will start by importing an GNCN-t1, which is an instantiation of the model
proposed in (Rao &amp; Ballard, 1999). To do so, we simply import a few modules
from the ngc-learn library, including a DataLoader, as follows:

```python
from ngclearn.utils.data_utils import DataLoader
```

the GNCN-t1 itself, as follows:

```python
from ngclearn.museum.gncn_t1 import GNCN_t1
```

and, finally, an argument Config object, some metrics, transformations,
and I/O tools, as follows:

```python
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.stat_utils as stat
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
```
