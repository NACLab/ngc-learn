# Overview

<!--
<b>ngc-learn</b> is a Python library for building, simulating, and analyzing
biomimetic computational models and arbitrary
predictive processing/coding models based on the neural generative
coding (NGC) computational framework. This toolkit is built on top of JAX and
is distributed under the 3-Clause BSD license.
-->

Advances made in research on artificial neural networks (ANNs) have led to many
breakthroughs in machine learning and beyond, resulting in the design of powerful
models that can categorize and forecast as well as agents that can play games and solve
complex problems. Behind these achievements is the backpropagation of errors
(or backprop) algorithm. Although elegant and powerful, a major long-standing
criticism of backprop has been its biological implausibility. In short, it is not
likely that the brain adjusts the synapses that connect the billions of neurons
that compose it in the way that backprop would prescribe.

Although ANNs are (loosely) inspired by our current understanding of the human brain,
the connections to the actual mechanisms that drive systems of natural neurons are
quite loose, at best. Although the question as to how the brain exactly conducts
credit assignment -- or the process of determining the contribution of each
and every neuron to the system's overall error on some task (the "blame game") -- is
still an open one, it would prove invaluable to have a flexible computational and software
framework that can facilitate the design and development of brain-inspired neural systems that
can also learn complex tasks. These tasks range from generative modeling to interacting and
manipulating dynamically-evolving environments. This would benefit researchers
in fields including, but not limited to, machine learning, (computational)
neuroscience, and cognitive science.

ngc-learn[^1], a Python simulation library, aims to fill the above need by
concretely instantiating neuronal dynamics and forms of
synaptic plasticity in the form of flexibly rearranged components and operations
to build arbitrary, modular, and complex biomimetic systems for research
in brain-inspired computing and neurocognitive modeling. More importantly, it is
designed to facilitate the design, development, and analysis of novel models of
neural computation and information processing, neuronal circuitry,
biologically-plausible credit assignment, and neuromimetic agents. Specifically,
ngc-learn implements a general schema for simulating biomimetic systems
characterized by differential equations, including ones based on
biophysical <i>spiking neuronal cells</i>.

The overarching goal of ngc-learn is to provide researchers and engineers with:
* a modular design that allows for the flexible creation, simulation, and analysis of
  neural systems and circuits under the framework of predictive processing;
* a powerful, approachable tool, written by and maintained by researchers and
experimenters directly studying and working to advance predictive processing and
biomimetic systems, meant to lower the barriers to entry to this field of research;
* a model museum that captures the essence of fundamental
and interesting predictive processing and other biomimetic models throughout
history, allowing for the study of and experimentation with classical and modern ideas.
<!--
* a ["model museum"](museum/model_museum) that captures the essence of fundamental
and interesting predictive processing and other biomimetic models throughout
history, allowing for the study of and experimentation with classical and modern ideas.
-->

The ngc-learn software framework was originally developed in 2019 by the Neural Adaptive
Computing (NAC) laboratory in Rochester Institute of Technology meant to serve as
an internal tool for predictive coding research (with earlier incarnations in the Scala
programming language, dating back to early 2017). It remains actively maintained by
and used for predictive processing and biomimetics research in the NAC lab
(see ngc-learn's mention/announcement in this
<a href="https://engineeringcommunity.nature.com/posts/the-neural-coding-framework-for-learning-generative-models">engineering blog post</a>).
We warmly welcome community contributions to this project. For details please check out our
[contributing guidelines](https://github.com/NACLab/ngc-learn/blob/main/CONTRIBUTING.md).

[^1]: The name `ngc-learn` stems from an important theory in neuroscience that served
as one of the library's first motivations to offer generalized research and
educational support for -- <i>predictive processing</i>, which posits that the
brain is largely a continual prediction engine, constantly hypothesizing the
state of its environment and updating its own internal mental model of it as
data is gathered. 
<!--
Moreover, prediction and correction happen at many levels or regions within the
brain -- clusters or groups of neurons in one region attempt to predict the state
of neurons at another region, forming a complex, somewhat hierarchical structure
that includes neurons which attempt to predict actual sensory input. Neurons within
this system adjust their internal activity values (as well the strengths of the
synapses that wire to them) based on how different their predictions were from
observed signals.
-->
The very first paradigm of neural computation that ngc-learn implemented and offered
general support for was a predictive coding framework known
as neural generative coding (NGC).

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
