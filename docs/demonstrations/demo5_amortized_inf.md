# Demo 5: Amortized Inference: Speeding Up the Settling Process

<i>NOTE: This demonstration is under construction and thus incomplete at the moment...</i>

Although fitting an NGC model (a GNCN) to a data sample is a rather straightforward
process, as we saw in the Demo 1 the underlying dynamics of the neural system
require performing K steps of an iterative settling (inference) process to find
suitable estimates of the latent neural state values. For the problem we have
investigated so far, this only required around 50 steps which is not too
expensive to simulate but for higher-dimensional, more complex problems, such
as modeling temporal data generating processes or learning from sparse signals
(as in the case of reinforcement learning), this settling process could
potentially start maxing out modest computational budgets.

There are, at least, two key paths to reduce the underlying computational
expense of the iterative settling process required by a predictive processing
NGC model:
1) exploit the layer-wise parallelism inherent to the NGC state and synaptic
update calculations -- since NGC models are not update-locked (the state predictions
and weight updates do not depend on one another) as deep neural
networks are, one could design a distributed algorithm where a group/system of
GPUs/CPUs synchronously (or asynchronously) compute(s) layer-wise predictions
and weight updates, and
2) reduce the number of settling steps by constructing a computation
process that infers the values of the latent states of a GNCN given a sensory
sample(s), ultimately serving as an intelligent initialization of the state values
instead of starting from zero vectors.
For this second way, approaches have ranged from ancestral sampling/projection,
as in deep Boltzmann machines [1] and as in for NGC systems formulated for active
inference [2], to learning (jointly with the generative model) a complementary
(neural) model in a process known as amortized inference, e.g., in sparse coding
the algorithm developed to do this was called predictive sparse decomposition [3].
Amortize means, in essence, to gradually reduce the initial cost of something
(whether it be an asset or activity) over a period.

In this demonstration, we will design a simple way to conduct amortized inference
to speed up the settling process of an NGC model, cutting down the number of
steps needed over time.


**References:**<br>
[1] Srivastava, Nitish, Ruslan Salakhutdinov, and Geoffrey Hinton. "Modeling
documents with a Deep Boltzmann Machine." Proceedings of the Twenty-Ninth
Conference on Uncertainty in Artificial Intelligence (2013). <br>
[2] Ororbia, A. G. & Mali, A. Backprop-free reinforcement learning with active
neural generative coding. In Proceedings of the AAAI Conference on Artificial
Intelligence Vol. 36 (2022).  <br>
[3] Kavukcuoglu, Koray, Marc'Aurelio Ranzato, and Yann LeCun. "Fast inference
in sparse coding algorithms with applications to object recognition."
arXiv preprint arXiv:1010.3467 (2010).
