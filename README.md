[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads)[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![Documentation Status](https://readthedocs.org/projects/ngc-learn/badge/?version=latest)](http://ngc-learn.readthedocs.io/en/latest/?badge=latest)[![DOI](https://zenodo.org/badge/483413212.svg)](https://zenodo.org/badge/latestdoi/483413212)

<img src="docs/images/ngc-learn-logo.png" width="300">

<b>ngc-learn</b> is a Python library for building, simulating, and analyzing arbitrary
predictive processing (or predictive coding) models based on the neural generative
coding (NGC) computational framework as well as other neurobiologically-motivated/grounded
systems. This toolkit is built on top of Tensorflow 2 and is distributed under the 3-Clause BSD license.

It is currently maintained by the
<a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing (NAC) laboratory</a>.

## <b>Documentation</b>

Official documentation, including tutorials, can be found
<a href="https://ngc-learn.readthedocs.io/en/latest/#">here</a>.

The official blog-post related to the source paper behind this software library
can be found
<a href="https://go.nature.com/3rgl1K8">here</a>.<br>
You can find the related paper (which also appears under the Editors' Highlights page
for Applied Physics and Mathematics), <a href="https://www.nature.com/articles/s41467-022-29632-7">right here</a>.

<!--The technical report going over the theoretical underpinnings of the
    NGC framework can be found here. TO BE RELEASED SOON. -->

## <b>Installation:</b>

<i>Setup:</i> Ensure that you have installed the following base dependencies in
your system. Note that this library was developed on Ubuntu 18.04 and tested on
Ubuntu(s) 16.04 and 18.04 (and should also work on Ubuntu 20.04).
ngc-learn requires:
1) Python (>=3.7)
2) Numpy (>=1.20.0)
3) Tensorflow 2.0.0, specifically, tensorflow-gpu>=2.0.0
4) scikit-learn (>=0.24.2) if using `ngclearn.density` (needed for the demo/tutorial
    files in `examples/`)
5) matplotlib (>=3.4.3) (for the demo/tutorial files in `examples/`)
6) networkx  (>=2.6.3) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)
7) pyviz (>=0.2.0) (currently optional but required if using `ngclearn.utils.experimental.viz_utils`)

(Which you can have setup/installed by running `$ pip3 install -r requirements.txt`.)

Once you have ensured that the above dependencies have been installed, you can then
run the setup script at the top of the <code>ngclearn</code> directory
to have the <code>ngclearn</code> package install on your system using the
following bash command:
<pre>
$ python setup.py install
</pre>

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


**Version:**<br>
0.4.0 <!-- -Alpha -->

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
