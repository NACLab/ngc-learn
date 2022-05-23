# Walkthrough 5: Amortized Inference

In this demonstration, we will design a simple way to conduct amortized inference
to speed up the settling process of an NGC model, cutting down the number of
steps needed overall. We will build a custom model, which we will call the
hierarchical ISTA model or "GNCN-t1-ISTA", and train it on the Olivetti database
of face images [4].
After going through this demonstration, you will:

1.  Learn how to construct a learnable inference projection graph to initialize
the states of an NGC system, facilitating amortized inference.
2.  Design a deep sparse coding model for modeling faces using the original
dataset used in [4] and visualize the acquired filters of the learned representation
system.

Note that the folders of interest to this demonstration are:
+ `walkthroughs/demo5/`: this contains the necessary simulation scripts
+ `walkthroughs/data`: this contains the zipped copy of the face image arrays

## Speeding Up the Settling Process with Amortized Inference

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
(neural) model, sometimes called a "recognition model", in a process known as amortized inference, e.g., in sparse coding the algorithm developed to do this was
called predictive sparse decomposition [3].
Amortize means, in essence, to gradually reduce the initial cost of something
(whether it be an asset or activity) over a period.

While there are many ways in which one could implement amortized inference, we
will focus on using ngc-learn's `ProjectionGraph` to construct a simple, learnable
recognition model.

## The Model: Hierarchical ISTA

We will start by first constructing the model we would like to learn. Specifically,
for this demonstration, we want to build a model for synthesizing human faces,
specifically those contained in the Olivetti faces database.

For this part of the demonstration, you will need to unzip the data contained
in `walkthroughs/data/faces.zip` (in the `walkthroughs/data/` sub-folder) to create
the necessary sub-folder which contains a single numpy array, `faces/dataX.npy`.
This data file contains the flattened vectors of `40` images of size `256 x 256`
pixels (pixel values have been normalized to the range of `[0,1]`), each
depicting a human face.
Two images sampled from the dataset (`dataX.npy`) are shown below:

```{eval-rst}
.. table::
   :align: center

   +------------------------------------------+------------------------------------------+
   | .. image:: ../images/demo4/face_img1.png | .. image:: ../images/demo4/face_img2.png |
   |   :scale: 30%                            |   :scale: 30%                            |
   |   :align: center                         |   :align: center                         |
   +------------------------------------------+------------------------------------------+
```

We will now construct the specialized model which we will call, in the context
of this demonstration, the "GNCN-t1-ISTA" (or "deep ISTA"). Specifically, we will
extend our sparse coding ISTA model from Demonstration \#4 to utilize an extra
layer of latent variables "above". Notably, we will use the soft-thresholding function,
which can be viewed as inducing a form a local lateral competition in the
latent activities to yield sparse representations, and apply to the two latent
state nodes of our system.

We start by first specifying the NGC system in design shorthand:

```
Node Name Structure:
z2 -(z2-mu1)-> mu1 ;e1; z1 -(z1-mu0-)-> mu0 ;e0; z0
```

where we see that our three-layer system consists of seven nodes in total, i.e.,
the three latent state nodes `z2`, `z1` and `z0`, the two mean prediction nodes
`mu1` and `mu0`, and the two error neuron nodes `e1` and `e0`. Note that, when
we build our recognition model later, our goal will be to infer good guess of the
initial values of the `z` compartment of the nodes `z1` and `z2` (with `z0` being
clamped to the input image patch `x`).

Inside of the provided `gncn_t1_ista.py`, we see how the core of the system
was put together with nodes and cables to create the hierarchical generative model:

```python
x_dim = # ... dimension of patch data ...
# ---- build a hierarchical ISTA model ----
K = 10
beta = 0.05
# general model configurations
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
thr_cfg = {"threshold_type" : "soft_threshold", "thr_lambda" : 5e-3}
# cable configurations
init_kernels = {"A_init" : ("unif_scale",1.0)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
pos_scable_cfg = {"type": "simple", "coeff": 1.0}
neg_scable_cfg = {"type": "simple", "coeff": -1.0}
constraint_cfg = {"clip_type":"forced_norm_clip","clip_mag":1.0,"clip_axis":1}

# set up system nodes
z2 = SNode(name="z2", dim=100, beta=beta, leak=0, act_fx="identity",
           integrate_kernel=integrate_cfg, threshold_kernel=thr_cfg)
mu1 = SNode(name="mu1", dim=100, act_fx="identity", zeta=0.0)
e1 = ENode(name="e1", dim=100)
z1 = SNode(name="z1", dim=100, beta=beta, leak=0, act_fx="identity",
           integrate_kernel=integrate_cfg, threshold_kernel=thr_cfg)
mu0 = SNode(name="mu0", dim=x_dim, act_fx="identity", zeta=0.0)
e0 = ENode(name="e0", dim=x_dim)
z0 = SNode(name="z0", dim=x_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)

# set up latent layer 2 to layer 1
z2_mu1 = z2.wire_to(mu1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
z2_mu1.set_constraint(constraint_cfg)
mu1.wire_to(e1, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
z1.wire_to(e1, src_comp="z", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
e1.wire_to(z2, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z2_mu1,"A^T"))
e1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

# set up latent layer 1 to layer 0
z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
z1_mu0.set_constraint(constraint_cfg)
mu0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
z0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A^T"))
e0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

# set up update rules and make relevant edges aware of these
z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), param=["A"])
z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])

# Set up graph - execution cycle/order
print(" > Constructing NGC graph")
model = NGCGraph(K=K, name="gncn_t1_ista")
model.set_cycle(nodes=[z2,z1,z0])
model.set_cycle(nodes=[mu1,mu0])
model.set_cycle(nodes=[e1,e0])
model.apply_constraints()
model.compile(batch_size=batch_size)
```

Notice that we have set the number of simulated settling steps `K` to be quite
small compared to the sparse coding models in Demonstration \#4, i.e., we have
drastically cut down the number of inference steps we required from `K = 300` to
`K = 10`, a highly desirable `96.6`\% decrease in computational cost (with
respect to number of settling steps). The key is that the recognition model
will learn to approximate the end-result of the settling process, and, over
the course of training to an image database, progressively improve its estimates
which will in turn better initialize the `NGCGraph` object's iterative inference.
Since the recognition model will continually chase the result of the ever-improving
settling process, we short-circuit the need for longer simulated settling processes
with the trade-off that our iterative inference will be a bit less accurate
in general (if the recognition model, which starts off randomly initialized,
provides bad starting points in the latent search space, then the settling
process will have to work harder to correct for the recognition model's
deficiencies).

## Constructing the Recognition Model

Building a recognition model for an NGC system is straightforward if we simply
treat it as an ancestral projection graph with the key exception that it is
"learnable". Specifically, we will randomly initialize an ancestral projection
graph that will compute initial "guesses" of the activity values of `z1` and `z2`
in our deep ISTA model. It helps to, as we did with the generative model, specify
the form of the recognition model in shorthand as follows:

```
Node Name Structure:
s0 -s0-s1-> s1 ; s1 -s1-s2-> s2
Note: s1; e1_i ; z1, s2; e2_i ; z2
Note: s0 = x  // (we clamp s0 to data)
```

where we emphasize the difference between the recognition model and the generative
model by labeling the recognition model's first and second latent layers as
`s1` and `s2`, respectively. Our recognition model's goal, as explained before,
will be to make its predicted value for `s1` match `z1` as well as
make its predicted value `s2` match `z2`, where `z1` and `z2` are the results
of the `NGCGraph` model's setting process that we designed above. This matching
task is emphasized by our shorthand's second line, where we see that the value
of `s1` will be compared to `z1` via the error node `e1_i` and `s2` will be
compared to `z2` via `e2_i`.

Unlike the previous projection graphs we have built in earlier walkthroughs,
our recognition model runs in the "opposite" direction of our generative model --
it takes in data and predicts initial values for the latent states while
the generative model predicts a value for the data given the latent states.
Together, the recognition and the generative model will learn to cooperate
in order to produce reasonable values for the latent states `z1` and `z2` that
could plausibly produce a given input image patch `z0 = x`.

To create the recognition model that will allow us to conduct amortized inference,
we write the following:

```python
# set up this NGC model's recognition model
inf_constraint_cfg = {"clip_type":"norm_clip","clip_mag":1.0,"clip_axis":0}
z2_dim = ngc_model.getNode("z2").dim
z1_dim = ngc_model.getNode("z1").dim
z0_dim = ngc_model.getNode("z0").dim

s0 = FNode(name="s0", dim=z0_dim, act_fx="identity")
s1 = FNode(name="s1", dim=z1_dim, act_fx="identity")
st1 = FNode(name="st1", dim=z1_dim, act_fx="identity")
s2 = FNode(name="s2", dim=z2_dim, act_fx="identity")
st2 = FNode(name="st2", dim=z2_dim, act_fx="identity")
s0_s1 = s0.wire_to(s1, src_comp="phi(z)", dest_comp="dz", cable_kernel=dcable_cfg)
s0_s1.set_constraint(inf_constraint_cfg)
s1_s2 = s1.wire_to(s2, src_comp="phi(z)", dest_comp="dz", cable_kernel=dcable_cfg)
s1_s2.set_constraint(inf_constraint_cfg)

# build the error neurons that examine how far off the inference model was
# from the final NGC system's latent activities
e1_inf = ENode(name="e1_inf", dim=z_dim)
s1.wire_to(e1_inf, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
st1.wire_to(e1_inf, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
e2_inf = ENode(name="e2_inf", dim=z_dim)
s2.wire_to(e2_inf, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
st2.wire_to(e2_inf, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)

# set up update rules and make relevant edges aware of these
s0_s1.set_update_rule(preact=(s0,"phi(z)"), postact=(e1_inf,"phi(z)"), param=["A"])
s1_s2.set_update_rule(preact=(s1,"phi(z)"), postact=(e2_inf,"phi(z)"), param=["A"])

sampler = ProjectionGraph()
sampler.set_cycle(nodes=[s0,s1,s2])
sampler.set_cycle(nodes=[st1,st2])
sampler.set_cycle(nodes=[e1_inf,e2_inf])
sampler.compile()
```

Now all that remains is to combine the recognition model with the generative model
to create the full system. Specifically, to tie the two components together, we
would write the following code:

```python
x = # ... sampled image patch (or batch of patches) ...
# run recognition model
readouts = sampler.project(
                clamped_vars=[("s0","z",x)],
                readout_vars=[("s1","z"),("s2","z")]
            )
s1 = readouts[0][2]
s2 = readouts[1][2]
# now run the settling process
readouts, delta = model.settle(
                    clamped_vars=[("z0","z", x)],
                    init_vars=[("z1","z",s1),("z2","z",s2)],
                    readout_vars=[("mu0","phi(z)"),("z1","z"),
                                  ("z2","z")],
                    calc_delta=True
                  )
x_hat = readouts[0][2]

# now compute the updates to the encoder given the current state of system
z1 = readouts[1][2]
z2 = readouts[2][2]
#z3 = readouts[3][2]
sampler.project(
    clamped_vars=[("s0","z",tf.cast(x,dtype=tf.float32)),
                  ("s1","z",s1),("s2","z",s2),
                  ("st1","z",z1),("st2","z",z2)]
)
r_delta = sampler.calc_updates()

# update NGC system synaptic parameters
opt.apply_gradients(zip(delta, model.theta))
# update recognition model synaptic parameters
r_opt.apply_gradients(zip(delta, sampler.theta))
```

The above code snippet would generally occur within your training loop (which
would be the same as the one in Demonstration \#4) and can be founded integrated
into the two key files provided for this demonstration, i.e., `sim_train.py`
and `gncn_t1_ista.py`. Note that the `gncn_t1_ista.py` further illustrates
how you can write a model that would fit within the general schema of ngc-learn's
Model Museum, which requires that NGC systems provide an API to their key
task-specific functions. `gncn_t1_ista.py` specifically implements all of the
code we developed above for the deep ISTA model and its corresponding
recognition model while `sim_train.py` is used to fit the model to the
Olivetti dataset you unzipped into the `walkthroughs/data/` directory.

To train our deep ISTA model, you should execute the following:

```console
python sim_train.py --config=sc_face/fit.cfg --gpu_id=0
```

which will simulate the training of a deep ISTA model on face image patches
for about `20` iterations. After this simulated process ends, you can then
run the visualization script we have created for you:

```console
$ python viz_filters.py --model_fname=sc_face/model0.ngc --output_dir=sc_face/ --viz_encoder=True
```

which will produce and save two visualizations in your `sc_face/` sub-directory,
one plot that depicts the learned bottom layer filters for the recognition
model and one for the deep ISTA model. You should see filter plots similar
to those presented below:

```{eval-rst}
.. table::
   :align: center

   +----------------------------------------------+----------------------------------------------+
   | .. image:: ../images/demo5/recog_filters.jpg | .. image:: ../images/demo5/model_filters.jpg |
   |   :scale: 40%                                |   :scale: 40%                                |
   |   :align: center                             |   :align: center                             |
   +----------------------------------------------+----------------------------------------------+
```

As we see, our NGC system has desirably learned low-level feature detectors
corresponding to "pieces" of human faces, such as lips, noses, eyes, and other
facial components. This was all learned only a few steps of simulated settling
(`K = 10`) utilizing our learned recognition model. Notice that the low-level
filters of the recognition model (the plot to the left) look similar to those
acquired by the generative model but are "simpler" or less distinguished/sharp.
This makes sense given that we designed our recognition model to "serve" the
generative model by providing an initialization of its latent states (or
"starting points" for the search for good latent states that generate the
input patches). It appears that the recognition model's facial feature detectors
are broad or less-detailed versions of those contained within our hierarchical
ISTA model.


## References
[1] Srivastava, Nitish, Ruslan Salakhutdinov, and Geoffrey Hinton. "Modeling
documents with a Deep Boltzmann Machine." Proceedings of the Twenty-Ninth
Conference on Uncertainty in Artificial Intelligence (2013). <br>
[2] Ororbia, A. G. & Mali, A. Backprop-free reinforcement learning with active
neural generative coding. In Proceedings of the AAAI Conference on Artificial
Intelligence Vol. 36 (2022).  <br>
[3] Kavukcuoglu, Koray, Marc'Aurelio Ranzato, and Yann LeCun. "Fast inference
in sparse coding algorithms with applications to object recognition."
arXiv preprint arXiv:1010.3467 (2010). <br>
[4] Samaria, Ferdinando S., and Andy C. Harter. "Parameterisation of a
stochastic model for human face identification." Proceedings of 1994 IEEE
workshop on applications of computer vision (1994).
