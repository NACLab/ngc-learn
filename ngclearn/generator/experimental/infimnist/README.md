The scripts contained in this folder, such as `infimnist_py`, have been crafted
to stably work with python3 and above.

The Python binding for the infinite MNIST dataset generator of L. Bottou
(see http://leon.bottou.org/projects/infimnist), is written in Cython.

**DEV NOTE:** We will be adding a real-time tensorflow generator to work with Keras
and Tensorflow 2 (TF2).

# Installation and Integration into ngc-learn

In order for ngc-learn to properly recognize the data generating process
under `ngclearn.generator.experimental.infimnist.infimnist`, you MUST
do the following:

1) You need to copy (or symlink) the `data/` directory from the original infimnist
project folder (available [here](http://leon.bottou.org/projects/infimnist))
into the root folder of this repo.

The files you specifically need inside of `data/` (that you can extract from the
source website's zip-file contents) are:
```
fields_float_1522x28x28.bin  
t10k-labels-idx1-ubyte         
train-images-idx3-ubyte
t10k-images-idx3-ubyte       
tangVec_float_60000x28x28.bin  
train-labels-idx1-ubyte
```

2) Next, build the cython extension with:

```
$ python setup.py build_ext -if
```

3) Finally, go up/back to the main ngc-learn directory `/ngc-learn/` and re-run
the primary library building script:

```
$ python setup.py install
```

and the InfiMNIST package should be successfully integrated into the ngc-learn
primary library package. If you do not complete the above directions, ngc-learn
will throw an error as it will not properly recognize the InfiMNIST sampler
in the `generator.experimental` module sub-directory.

# Example Usage
The following code gives the first test example (0), the first training example
(10000) and its first transformation (70000). The indexing logic follows that
of the original infimnist binary described [here](http://leon.bottou.org/projects/infimnist).
ngc-learn has built internally a wrapper "generator" to convert the original
infinite mnist into an on-the-fly (mini-batch) sampler/stochastic data generating
process.

```python
from ngclearn.generator.experimental.infimnist.infimnist import InfiMNIST
import matplotlib.pyplot as plt

n_iter = 10 # bound your computation by a maximum number of iterations to simulate

sampler = InfiMNIST(batch_size=8)

for t in range(n_iter):
    x, y = sampler.sample()

    # Update NGC agent given x, y
    # WRITEME: agent code goes here...

    # You can also plot images using matplotlib and save to disk if you like
    import matplotlib.pyplot as plt
    plt.imshow(x[0].reshape(28,28))
    plt.title('label: {}'.format(y[0]))
    plt.savefig("sample{}.jpg".format(t))
```

**DEV NOTE:** Future versions of InfiMNIST will offer support for tf.records.
