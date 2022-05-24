# The Model Museum

Predictive processing has undergone many important developments over the decades,
dating back to Hermann von Helmholtz's theory of  “unconscious inference”
in perception which itself operationalized the ideas of the 18th century philosopher
Immanuel Kant. It has risen as a promising theoretical and mathematical model
of various aspects of neural circuitry in computational neuroscience, serving
as one embodiment of the Bayesian brain hypothesis, and has been shown to be a
powerful computational modeling tool for cognitive science and
statistical/machine learning. Many different architectures/systems, often designed to serve
one or a few particular modeling purposes, have been (and still are being) proposed.
Given this, one of ngc-learn's aims is to capture an approximate snapshot of as many
of these architectures/ideas as possible.

Given the generality of the NGC computational framework [1], many
flavors of predictive processing can be recovered/derived and it is within
ngc-learn's Model Museum that we intend to model and preserve variant
models (as they historically have been and currently are being created). This
allows current and future scientists, engineers, and enthusiasts to examine
these models, much as one would curiously examine exhibits. such as paintings
or preserved mechanical inventions and technological artifacts, at a museum.
The Model Museum also provides an opportunity for those working in the domain
of predictive processing to publish their successful structures/ideas that are
presented in publications and/or tested applications (contact us if you have a
particular published or representative predictive processing model that you
would like exhibited and to be integrated into the Model Museum for the benefit
of the community).
In parallel, since ngc-learn is an evolving library, we will be working to
curate and update the museum with representative models over time
and several are already under development/testing (stay tuned for their release
across software releases/patches/edits).

As mentioned above, NGC predictive processing models have historically been
designed to serve particular purposes, thus we wrap their underlying NGC graphs
in an agent structure that provides particular documented convenience functions
that allow the user/modeler to interact with such models according to their
intended purpose/use. For example, a published/public NGC model that was
developed to classify data will offer functionality for categorization in a
relevant prediction routine while another one that was created to operate as
a generative/density estimator will sport a routine(s) for sampling/synthesization.

Current models that we have implemented in the Model Museum so far include:
1. GNCN-t1/Rao - the model proposed in (Rao &amp; Ballard, 1999) [2]
2. GNCN-t1-Sigma/Friston - the model proposed in (Friston, 2008) [3]
3. GNCN-PDH - the model proposed in (Ororbia &amp; Kifer, 2022) [1]
4. GNCN-t1-FFM - the model developed in (Whittington &amp; Bogacz, 2017) [4]
5. GNCN-t1-SC - the model proposed in (Olshausen &amp; Field, 1996) [5]
6. Harmonium - the model developed in (Smolensky, 1986; Hinton 1999) [6] [7]

(If there is a model you think should be exhibited/integrated into the Model
Museum, and/or would like to contribute, please write us at ago@cs.rit.edu or
raise a github issue.)


**References:** <br>
[1] Ororbia, A., and Kifer, D. The neural coding framework for learning
generative models. Nature Communications 13, 2064 (2022). <br>
[2] Rao, Rajesh PN, and Dana H. Ballard. "Predictive coding in the visual cortex:
a functional interpretation of some extra-classical receptive-field effects."
Nature neuroscience 2.1 (1999): 79-87. <br>
[3] Friston, Karl. "Hierarchical models in the brain." PLoS Computational
Biology 4.11 (2008): e1000211. <br>
[4] Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262. <br>
[5] Olshausen, B., Field, D. Emergence of simple-cell receptive field properties
by learning a sparse code for natural images. Nature 381, 607–609 (1996). <br>
[6] Hinton, Geoffrey E. "Training products of experts by maximizing contrastive
likelihood." Technical Report, Gatsby computational neuroscience unit (1999). <br>
[7] Smolensky, P. "Information Processing in Dynamical Systems: Foundations of
Harmony Theory." Parallel distributed processing: explorations in the
microstructure of cognition 1 (1986).
