# Demo 3: Creating an NGC Classifier

<i>NOTE: This demonstration is under construction and thus incomplete at the moment...</i>

In this demonstration, we will learn how to create a classifier based on NGC.
After going through this demonstration, you will:

1.  Learn how to use a simple projection graph and the `extract()` routine to
initialize the simulated settling process of an NGC model.
2.  Craft and simulate an NGC model that can directly classify the image
patterns in the MNIST database (from Demonstration \# 1), reproducing similar
results to what was reported in (Whittington &amp; Bogacz, 2017).

Note that the folder of interest to this demonstration is:
+ `examples/demo3/`: this contains the necessary simulation script

## Using an Ancestral Projection Graph to Initialize Iterative Inference

We will start by first discussing an important use-case of the `ProjectionGraph` --
to initialize the simulated iterative inference process of an `NGCGraph`. This is
contrast to the use-case we saw in the last two demonstrations where we used the
ancestral projection graph as a post-training tool, which allowed us to draw
samples from the underlying directed generative models we were fitting. This time,
we will leverage the power of an ancestral projection graph to serve as a
simple, progressively improving model of initial conditions for an iterative inference
process.

...

## Learning a Classifier

In addition to serving as a powerful initial conditions model for the NGC system
that we will build for the demonstration, we will also exploit an important by-product
of the discriminative learning our code will simulate -- the projection graph
also offers a fast feedforward classifier. This will give us a fast way to conduct
test-time inference when tracking model performance/generalization as well as a final
classifier that we can directly use without resorting to iterative inference (for
example, one could extract the synaptic weight matrices inside the ancestral
projection graph, serialize them to disk, and place them inside a multi-layer
perceptron structure one builds with pure Tensorflow or Pytorch in a downstream
application).

...
