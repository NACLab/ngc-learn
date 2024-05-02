[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads)[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![Documentation Status](https://readthedocs.org/projects/ngc-learn/badge/?version=latest)](http://ngc-learn.readthedocs.io/en/latest/?badge=latest)[![DOI](https://zenodo.org/badge/483413212.svg)](https://zenodo.org/badge/latestdoi/483413212)

<img src="docs/images/ngc-learn-logo.png" width="300">

<b>ngc-learn</b> is a Python library for building, simulating, and analyzing
biomimetic systems, neurobiological agents, spiking neuronal networks,
predictive coding circuitry, and models that learn via biologically-plausible
forms of credit assignment. This simulation toolkit is built on top of JAX and is
distributed under the 3-Clause BSD license.

It is currently maintained by the
<a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing (NAC) laboratory</a>.

## <b>Documentation</b>

Official documentation, including tutorials, can be found
<a href="https://ngc-learn.readthedocs.io/en/latest/#">here</a>. The model museum repo,
which implements several historical models, can be found
<a href="https://github.com/NACLab/ngc-museum">here</a>.

The official blog-post related to the source paper behind this software library
can be found
<a href="https://go.nature.com/3rgl1K8">here</a>.<br>
You can find the related paper <a href="https://www.nature.com/articles/s41467-022-29632-7">right here</a>, which
was selected to appear in the Nature <i>Neuromorphic Hardware and Computing Collection</i> in 2023 and was
chosen as one of the <i>Editors' Highlights for Applied Physics and Mathematics</i> in 2022.

<!--The technical report going over the theoretical underpinnings of the
    NGC framework can be found here. TO BE RELEASED SOON. -->

## Installation

### Dependencies

ngc-learn requires:
1) Python (>=3.10)
2) NumPy (>=1.26.0)
3) SciPy (>=1.7.0)
4) ngcsimlib (>=0.2.b1), (visit official page <a href="https://github.com/NACLab/ngc-sim-lib">here</a>)
5) JAX (>= 0.4.18) (to enable GPU use, make sure to install one of the CUDA variants)
<!--
5) scikit-learn (>=1.3.1) if using `ngclearn.utils.density`
6) matplotlib (>=3.4.3) if using `ngclearn.utils.viz`
6) networkx  (>=2.6.3) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)
7) pyviz (>=0.2.0) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)
-->

---
ngc-learn 1.0.beta0 and later require Python 3.10 or newer as well as ngcsimlib >=0.2.b2.
ngc-learn's plotting capabilities (routines within `ngclearn.utils.viz`) require
Matplotlib (>=3.8.0) and imageio (>=2.31.5) and both plotting and density estimation
tools (routines within ``ngclearn.utils.density``) will require Scikit-learn (>=0.24.2).
Many of the tutorials will require Matplotlib (>=3.8.0), imageio (>=2.31.5), and Scikit-learn (>=0.24.2).
<!-- (Note: if using the `_generate_patch_set()` within the
image patching utilities, then Patchify will be needed).-->

### User Installation

<i>Setup</i>: The easiest way to install ngc-learn (CPU version) is through <code>pip</code>:
<pre>
$ pip install ngclearn
</pre>

The documentation includes more detailed
<a href="https://ngc-learn.readthedocs.io/en/latest/installation.html">installation instructions</a>.
Note that this library was developed on Ubuntu 20.04 and tested on Ubuntu(s) 18.04 and 20.04.

If the installation was successful, you should see the following if you test
it against your Python interpreter, i.e., run the <code>$ python</code> command
and complete the following sequence of steps as depicted in the screenshot below
(you should see at the bottom of your output something akin to the
right major and minor version of ngc-learn):

```console
Python 3.11.4 (main, MONTH  DAY YEAR, TIME) [GCC XX.X.X] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import ngclearn
>>> ngclearn.__version__
'1.0b3'
```

<i>Note:</i> For access to the previous Tensorflow-2 version of ngc-learn (of
which we no longer support), please visit the repo for
<a href="https://github.com/NACLab/ngc-learn-legacy"><i>ngc-learn-legacy</i></a>.

## <b>Attribution:</b>

If you use this code in any form in your project(s), please cite its source
paper (as well as ngc-learn's official software citation):
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
$ git clone https://github.com/NACLab/ngc-learn.git
</pre>

If you are working on and developing with ngc-learn pulled from the github
repo, then run the following command to set up an editable install:
<pre>
$ python install -e .
</pre>

**Version:**<br>
1.0.4-Beta <!-- -Alpha -->

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
