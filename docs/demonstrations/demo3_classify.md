# Demo 3: Creating an NGC Classifier

<i>NOTE: This demonstration is under construction and thus incomplete at the moment...</i>

In this demonstration, we will learn how to create a classifier based on NGC.
After going through this demonstration, you will:

1.  Learn how to use a simple projection graph and the `extract()` routine to
initialize the simulated settling process of an NGC model.
2.  Craft and simulate an NGC model that can directly classify the image
patterns in the MNIST database (from Demonstration \# 1), producing results
comparable to what was reported in (Whittington &amp; Bogacz, 2017).

Note that the folder of interest to this demonstration is:
+ `examples/demo3/`: this contains the necessary simulation script
+ `examples/data`: this contains a zipped copy of the MNIST database arrays

## Using an Ancestral Projection Graph to Initialize Iterative Inference

We will start by first discussing an important use-case of the `ProjectionGraph` --
to initialize the simulated iterative inference process of an `NGCGraph`. This is
contrast to the use-case we saw in the last two demonstrations where we used the
ancestral projection graph as a post-training tool, which allowed us to draw
samples from the underlying directed generative models we were fitting. This time,
we will leverage the power of an ancestral projection graph to serve as a
simple, progressively improving model of initial conditions for an iterative inference
process.

To illustrate the above use-case, we will focus on crafting an NGC model for
discriminative learning (as opposed to the generative learning models we built
Demonstrations \# 1 and \#2). Before working with a concrete application, as we
will do in the next section, let us just focus on crafting the NGC architecture
of the classifier as well as its ancestral projection graph.

Working with nodes and cables (see [the last demonstration for details](../demonstrations/demo2_create_ngc.md)),
we will build a simple hierarchical system that adheres to the following NGC shorthand:

```
Node Name Structure:
z2 -(z1-mu1)-> mu1 ;e1; z1 -(z1-mu0-)-> mu0 ;e0; z0
Note that z3 = x and z0 = y, which yields a classifier
```




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

Specifically, we will now build a supervised
classifier using the labels that come with the processed MNIST dataset, in `mnist.zip`,
that you worked with in Demonstration \# 1).

...
