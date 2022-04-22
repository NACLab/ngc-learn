# Overview

<b>ngc-learn</b> is a Python library for building, simulating, and analyzing
arbitrary predictive processing models based on the neural generative coding (NGC)
computational framework. This toolkit is built on top of Tensorflow 2 and is
distributed under the 3-Clause BSD license.

Advances made in research on artificial neural networks (ANNs) have led to many
breakthroughs in machine learning and beyond, resulting in the design of powerful
models that can categorize and forecast as well as agents that can play games and solve
complex problems. Behind these achievements is the backpropagation of errors
(or backprop) algorithm. Although elegant and powerful, a major long-standing
criticism of backprop has been its biological implausibility. In short, it is not
likely that billions of neurons that compose the human brain adjust the synapses
that connect in the way that backprop would prescribe.

Although ANNs are (loosely) inspired by our current understanding of the human brain,
the connections to the actual mechanisms that drive systems of natural neurons are
quite loose, at best. Although the question as to how the brain exactly conducts
credit assignment -- or the process of determining the contribution of each
and every neuron to overall error on some task (the "blame game") -- is still an
open one, it would prove invaluable to have a flexible computational framework that
can facilitate the design and development of brain-inspired neural systems that
can also learn complex tasks, ranging from generative modeling to interacting and
manipulating dynamically-evolving environments. This would benefit researchers
in fields including, but not limited to, machine learning, (computational)
neuroscience, and cognitive science.

ngc-learn aims to fill the above need by concretely instantiating an important
theory in neuroscience known as <i>predictive processing</i>, positing that the brain
is largely a continual prediction engine, constantly predicting the state of its
environment and updating its own internal mental model of it. Moreover, prediction
and correction happens at many levels or regions within the brain -- clusters or
groups of neurons in one region attempt to predict the state of neurons at another
region, forming a complex somewhat hierarchical structure that includes neurons
which attempt to predict actual sensory input. Neurons within this system adjust
their internal activity values as well the strengths of the synapses that wire
to them based on how different their predictions were from observed signals.
Concretely, ngc-learn implements a general predictive processing framework known
as neural generative coding (NGC).

The overarching goal of ngc-learn is to provide researchers and engineers with:
* a modular design that allows for the flexible design, simulation, analysis of
  neural systems fundamentally built and driven by predictive processing;
* a powerful, approachable tool, written by and maintained by researchers and
experimenters directly studying and working to advance predictive processing,
meant to lower the barriers to entry to this field of research;
* a "model museum" that captures the essence of fundamental and interesting
predictive processing models and algorithms throughout history, allowing for the
study of and experimentation with classical and modern systems.

The ngc-learn software framework was originally developed in 2019 by the Neural Adaptive
Computing (NAC) laboratory in Rochester Institute of Technology meant as an internal
tool for predictive processing research (with earlier incarnations in the Scala
programming language, dating back to early 2017). It remains actively maintained
and used for predictive processing research in NAC.
We welcome community contributions to this project. For details please check out our contributing guidelines (<i>coming soon!</i>).

<!--
This release of ngc-learn contains three predictive processing models, X types of
nodes, Y types of cables, and Z density estimators. It also offers a modular design of NGC systems
for building new/novel and general architectures and models. We highlight these primary features below:
-->

## Citation
Please cite ngc-learn's source/core paper if you use this framework in your publications:
```
@article{Ororbia2022,
  author={Ororbia, Alexander and Kifer, Daniel},
  title={The neural coding framework for learning generative models},
  journal={Nature Communications},
  year={2022},
  month={Apr},
  day={19},
  volume={13},
  number={1},
  pages={2064},
  issn={2041-1723},
  doi={10.1038/s41467-022-29632-7},
  url={https://doi.org/10.1038/s41467-022-29632-7}
}
```
