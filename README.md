[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads)[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![Documentation Status](https://readthedocs.org/projects/ngc-learn/badge/?version=latest)](http://ngc-learn.readthedocs.io/en/latest/?badge=latest)[![DOI](https://zenodo.org/badge/483413212.svg)](https://zenodo.org/badge/latestdoi/483413212)

<img src="docs/images/ngc-learn-logo.png" width="300">

<b>ngc-learn</b> is a Python library for building, simulating, and analyzing biomimetic systems, neurobiological
agents, spiking neuronal systems, and predictive coding models (predictive processing theory) based on the neural generative
coding (NGC) computational framework. This toolkit is built on top of JAX and is distributed under the 3-Clause BSD license.

<b>NOTICE:</b> This is currently the JAX branch of ngc-learn; we offer <i>no guarantees</i> as to whether or not this particular branch works at the moment (until we finish the ngc-learn update to officially release version 1.0.0). For access to the previous working Tensorflow-2 version, please visit the repo for <a href="https://github.com/NACLab/ngc-learn-legacy"><i>ngc-learn-legacy</i></a>.

It is currently maintained by the
<a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing (NAC) laboratory</a>.

## <b>Documentation</b>

Official documentation, including tutorials, can be found
<a href="https://ngc-learn.readthedocs.io/en/latest/#">here</a>.

The official blog-post related to the source paper behind this software library
can be found
<a href="https://go.nature.com/3rgl1K8">here</a>.<br>
You can find the related paper <a href="https://www.nature.com/articles/s41467-022-29632-7">right here</a>, which
was selected to appear in the Nature <i>Neuromorphic Hardware and Computing Collection</i> in 2023 and was
chosen as one of the <i>Editors' Highlights for Applied Physics and Mathematics</i> in 2022.

<!--The technical report going over the theoretical underpinnings of the
    NGC framework can be found here. TO BE RELEASED SOON. -->

## <b>Installation:</b>

<i>Setup:</i> To install ngc-learn, you can run (at the top-level of the
the <code>ngclearn</code> directory) the following bash command:
<pre>
$ python install .
</pre>
which will ensure that all the required base dependencies are installed in
your system. Note that this library was developed on Ubuntu 20.04 and tested on
Ubuntu(s) 18.04 and 20.04.
ngc-learn requires:
1) ngclib (>=0.2.0), (for installation, visit <a href="https://github.com/NACLab/ngc-lib">here</a>)
2) Python (>=3.9)
3) Numpy (>=1.26.0)
4) JAX (>= 0.4.16) (to enable GPU use, make sure to install one of the CUDA variants)
5) scikit-learn (>=1.3.1) if using `ngclearn.density` (needed for the demo/tutorial
    files in `examples/`)
6) matplotlib (>=3.4.3) (for the demo/tutorial files in `examples/`)
<!--
6) networkx  (>=2.6.3) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)
7) pyviz (>=0.2.0) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)
-->

<i>Note:</i> Running the above pip install will automatically install the CPU
version of JAX. If you want to use the GPU version instead, make sure to,
before running the above, to install JAX via the correct pip command
with the proper CUDA flags (depending on which CUDA is configured for your system)
as per their
<a href="https://jax.readthedocs.io/en/latest/installation.html">installation instructions</a>.

<!--
(If you want to set up/install dependencies a priori, try running
`$ pip install -r requirements.txt` first before pip installing ngc-learn.)
-->

If the installation was successful, you should see the following if you test
it against your Python interpreter, i.e., run the <code>$ python</code> command
and complete the following sequence of steps as depicted in the screenshot below:<br>
<img src="docs/images/test_ngclearn_install.png" width="512">

## <b>Attribution:</b>

If you use this code in any form in your project(s), please cite its source
paper:
<pre>
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
</pre>

## <b>Development:</b>

We warmly welcome community contributions to this project. For details on how to
make a contribution to ngc-learn, please see our
[contributing guidelines](CONTRIBUTING.md).

<b>Source Code</b>
You can check/pull the latest source code for this library via:
<pre>
$ git clone https://github.com/ngc-learn/ngc-learn.git
</pre>

If you are working on and developing ngc-learn, then run the following command:
<pre>
$ python install -e . # sets up an editable install
</pre>

**Version:**<br>
1.0.0-Alpha <!-- -Alpha -->

Author:
Alexander G. Ororbia II<br>
Director, Neural Adaptive Computing (NAC) Laboratory<br>
Rochester Institute of Technology, Department of Computer Science

## <b>Copyright:</b>

Copyright (C) 2021 The Neural Adaptive Computing Laboratory - All Rights Reserved<br>
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this software.<br>
If not, please [email us](mailto:ago@cs.rit.edu)
