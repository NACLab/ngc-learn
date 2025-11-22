# Density Modeling and Analysis 

NGC-Learn offers some support for density modeling/estimation, which can be particularly useful in analyzing how internal properties of neuronal models organize (e.g., how the distributed representations of a model might cluster into distinct groups/categories) or to draw samples from the underlying generative model implied by a particular neuronal structure (e.g., sampling a predictive coding generative model). Particularly, within `ngclearn.utils.density`, one can find implementations of mixture models, e.g., a Gaussian mixture model (GMM), that might be implied to carry out such tasks. In this small lesson, we will demonstrate how to set up a GMM, fit it to some synthetic latent code data, and plot out the distribution it learns overlaid over the data samples as well as examine the kinds of patterns one may sample from the learnt GMM.

## Setting Up a Gaussian Mixture Model

Let's say you have a two-dimensional dataset of neural code vectors collected from another model you have simulated -- here, we will artificially synthesize this kind of data in this lesson from an "unobserved" trio of multivariate Gaussians (as was done in the t-SNE tutorial) -- and that, furthermore, you wanted to fit a GMM to these codes to later on sample from their underlying multi-modal distribution.

The following Python code will employ a GMM density estimator for you (including setting up the data generator):

```python
from jax import numpy as jnp, random
from ngclearn.utils.density.gmm import GMM ## pull out the mixture model density estimator

def gen_data(dkey, n_samp_per_mode): ## data generator (or proxy stochastic data generating process)
    scale = 0.3
    mu1 = jnp.asarray([[2.1, 3.2]]) * scale
    cov1 = jnp.eye(mu1.shape[1]) * 0.78 * scale * 0.5
    mu2 = jnp.asarray([[-2.8, 2.0]]) * scale
    cov2 = jnp.eye(mu2.shape[1]) * 0.52 * scale * 0.5
    mu3 = jnp.asarray([[1.2, -2.7]]) * scale
    cov3 = jnp.eye(mu3.shape[1]) * 1.2 * scale * 0.5
    params = (mu1,cov1 ,mu2,cov2,mu3,cov3)

    dkey, *subkeys = random.split(dkey, 7)
    samp1 = random.multivariate_normal(subkeys[0], mu1, cov1, shape=(n_samp_per_mode,))
    samp2 = random.multivariate_normal(subkeys[0], mu2, cov2, shape=(n_samp_per_mode,))
    samp3 = random.multivariate_normal(subkeys[0], mu3, cov3, shape=(n_samp_per_mode,))
    X = jnp.concatenate((samp1, samp2, samp3), axis=0)
    y1 = jnp.ones((n_samp_per_mode, 3)) * jnp.asarray([[1., 0., 0.]])
    y2 = jnp.ones((n_samp_per_mode, 3)) * jnp.asarray([[0., 1., 0.]])
    y3 = jnp.ones((n_samp_per_mode, 3)) * jnp.asarray([[0., 0., 1.]])
    lab = jnp.concatenate((y1, y2, y3), axis=0) ## one-hot codes
    return X, lab, params

## set up the GMM density estimator
key = random.PRNGKey(69)
dkey, _ = random.split(key, 2)
X, y, params = gen_data(key, n_samp_per_mode=200) #400)

n_iter = 30 
n_components = 3
model = GMM(K=n_components, max_iter=n_iter, key=dkey)
model.init(X) ## initailize the GMM to dataset X
```

The above will construct a GMM with three components (or latent variables of its own) and be configured to use a maximum of `30` iterations to fit itself to data. Note that the call to `init()` will "shape" the GMM according to the dimensionality of the data and pre-initialize its parameters (i.e., choosing random data vectors to initialize its means). 

To fit the GMM itself to your dataset `X`, you will then write the following: 

```python
## estimate GMM parameters over dataset via E-M
model.fit(X, tol=1e-3, verbose=True) ## set verbose to `False` to silence the fitting process
```

which should print to I/O something akin to: 

```console
0: Mean-diff = 0.8029823303222656
1: Mean-diff = 0.1899024397134781
2: Mean-diff = 0.18127720057964325
3: Mean-diff = 0.15023663640022278
4: Mean-diff = 0.13917091488838196
5: Mean-diff = 0.10519692301750183
6: Mean-diff = 0.05732756853103638
7: Mean-diff = 0.03420640528202057
8: Mean-diff = 0.01907791942358017
9: Mean-diff = 0.009763183072209358
10: Mean-diff = 0.004887263756245375
11: Mean-diff = 0.0024237236939370632
12: Mean-diff = 0.0011952449567615986
13: Mean-diff = 0.0005875130300410092
Converged after 14 iterations.
```

In the above instance, notice that our GMM converged early, reaching a good log likelihood in `14` iterations. We can further calculate our model's log likelihood over the dataset `X` with the following in-built function:

```python
# Calculate the GMM log likelihood 
_, logPX = model.calc_log_likelihood(X) ## 1st output is log-lieklihood per data pattern
print(f"log[p(X)] = {logPX} nats")
```

which will print out the following:

```console
log[p(X)] = -423.30889892578125 nats
```

Now, to visualize if our GMM actually capture the underlying multi-modal distribution of our dataset, we may visualize the final GMM with the following plotting code: 

```python
import matplotlib.pyplot as plt
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 100), jnp.linspace(y_min, y_max, 100))
Xspace = jnp.c_[xx.ravel(), yy.ravel()]
Z, _ = model.calc_log_likelihood(Xspace) # Get log likelihood (LL)
Z = -Z ## flip sign of LL (to get negative LL)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c="blue", s=10, alpha=0.7, label='Latent Codes')
plt.contour(xx, yy, Z, levels=jnp.logspace(0, 2, 12), cmap='viridis', alpha=0.8)
plt.colorbar(label='Negative Log Likelihood')

plt.title('GMM Distribution Plot')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.grid(True)
plt.savefig("gmm_fit.jpg") #plt.show()

plt.close()
```

which should produce a plot similar to the one below: 

<img src="../../images/tutorials/neurocog/gmm_fit.jpg" width="400" />


To draw samples from our fitted/learnt GMM, we may next call its in-built synthesizing routine as follows:

```python
## Examine GMM samples
Xs = model.sample(n_samples=200 * 3) ## draw 600 samples from fitted GMM
```

and then visualize the collected batch of samples with the following plotting code: 

```python

plt.figure(figsize=(8, 6))
plt.scatter(Xs[:, 0], Xs[:, 1], c="green", s=10, alpha=0.7, label='Sample Points')
plt.contour(xx, yy, Z, levels=jnp.logspace(0, 2, 12), cmap='viridis', alpha=0.8)
plt.colorbar(label='Negative Log-Likelihood')
plt.title('GMM Samples')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.grid(True) #plt.show()
plt.savefig("gmm_samples.jpg")

plt.close()
```

which will produce a plot similar to the one below: 

<img src="../../images/tutorials/neurocog/gmm_samples.jpg" width="400" />

Notice that the green-colored data points roughly adhere to the contours of the GMM distribution and look much like the original (blue-colored) dataset we collected. In this example scenario, we see that we can successfully learn the density of our latent code dataset, facilitating some level of downstream distributional analysis and generative model sampling.
