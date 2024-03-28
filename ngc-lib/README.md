[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads)[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![DOI](https://zenodo.org/badge/734040498.svg)](https://zenodo.org/doi/10.5281/zenodo.10888210)

# NGC-Lib: Support Library for NGC-Learn

<b>ngc-lib</b> is the support library and central dependency for
<i><a href="https://github.com/NACLab/ngc-learn/">ngc-learn</a></i>, a library
designed for computational neuroscience and cognitive neuroscience research.
While ngc-learn contains the JAX-implemented routines and any supporting C
code, ngc-lib is a pure Python package, primarily meant for providing the
machinery, routines, and utilities that facilitate the general (abstract)
simulation of complex adaptive systems made up of dynamical components. For
information, including anything related to usage instructions and details,
please refer to the ngc-learn README:
https://github.com/NACLab/ngc-learn/.

This package is is distributed under the 3-Clause BSD license. It is currently
maintained by the
<a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing
(NAC) laboratory</a>.

## <b>Installation:</b>

<i>Setup:</i> Ensure that you have installed the following base dependencies in
your system. Note that this library was developed and tested on
Ubuntu 22.04.3 LTS. ngc-lib requires: `Python (>=3.10)`.

Once you have ensured that the appropriate Python is installed, you can then
have the <code>ngclib</code> package installed on your system using the
following bash command:
<pre>
$ pip install .
</pre>
or, if you are doing development, then do an editable install via:
<pre>
$ pip install --editable . # OR pip install -e .
</pre>

**Version:**<br>
0.2.0 <!-- -Alpha -->

Authors:
William Gebhardt, Alexander G. Ororbia II<br>
Neural Adaptive Computing (NAC) Laboratory<br>
Rochester Institute of Technology, Department of Computer Science

## <b>Copyright:</b>

Copyright (C) 2023 The Neural Adaptive Computing Laboratory - All Rights Reserved<br>
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this software.<br>
If not, please [email us](mailto:ago@cs.rit.edu)
