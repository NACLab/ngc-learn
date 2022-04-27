# The Model Museum

Predictive processing has undergone many important developments over the decades,
developing as a promising mathematical model of neural circuitry in computational
neuroscience, serving as one embodiment of the Bayesian brain hypothesis, as
well as a powerful computational processing model for cognitive science and
statistical/machine learning. Many different models, often designed to serve
one or a few particular modeling purposes, have been proposed (and still are)
and one of ngc-learn's aims is to capture an approximate snapshot of as many as
possible.

Given the generality of the NGC computational framework [1], many
flavors of predictive processing can be recovered/derived and it is within
ngc-learn's so-called Model Museum that we intend to model and preserve variant
models (as they historically have been and currently are being created). This
allows current and future scientists, engineers, and enthusiasts to examine
these models, much as one would curiously examine exhibits (such paintings
or preserved mechanical inventions/technology) in a museum.
The Model Museum also provides an opportunity for those working in the domain
of predictive processing to publish their successful structures/ideas that are
presented in publications and/or tested applications (contact us if you have a
particular published or representative predictive processing model that you
would like exhibited and to integrate into the Model Museum).
In parallel, as ngc-learn will be an evolving library, we will be working to
curate and update the museum with representative models over time
and several are already under development/testing (stay tuned for their release
across software releases/patches/edits).

As mentioned above, NGC predictive processing models have historically been
designed to serve particular purposes, thus we wrap their underlying NGC graphs
in an agent structure that provides particular documented convenience functions
allowing the user/modeler to interact with such models according to their
intended purpose/use. For example, a published/public NGC model that was
developed to classify data will offer functionality for categorization in a
relevant prediction routine while another one that was created to operate as
a generative/density estimator will sport a routine(s) for sampling/synthesization.

Current models that we have implemented in the Model Museum so far are:
1. GNCN-t1/Rao - the model proposed in (Rao &amp; Ballard, 1999) [2]
2. GNCN-t1-Sigma/Friston - the model proposed in (Friston, 2008) [3]
3. GNCN-PDH - the model proposed in (Ororbia &amp; Kifer, 2022) [1]
4. GNCN-t1-FFM - the model proposed in (Whittington &amp; Bogacz, 2017) [4]


**References:** <br>
[1] Ororbia, A., and Kifer, D. The neural coding framework for learning
generative models. Nature Communications 13, 2064 (2022). <br>
[2] Rao, Rajesh PN, and Dana H. Ballard. "Predictive coding in the visual cortex:
a functional interpretation of some extra-classical receptive-field effects."
Nature neuroscience 2.1 (1999): 79-87. <br>
[3] Friston, Karl. "Hierarchical models in the brain." PLoS Computational
Biology 4.11 (2008): e1000211.
[4] Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.
