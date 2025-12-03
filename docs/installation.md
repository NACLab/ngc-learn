# Installation

**ngc-learn** officially supports Linux on Python 3. It can be run with or without a GPU.

<i>Setup:</i> <a href="https://github.com/NACLab/ngc-learn">NGC-Learn</a>, in its entirety (including its supporting utility sub-packages), requires that you ensure that you have installed the following base dependencies in your system. Note that this library was developed and tested on Ubuntu 22.04 (with much earlier versions on Ubuntu 18.04/20.04). 
Specifically, NGC-Learn requires:
* Python (>=3.10)
* ngcsimlib (>=2.0.0), (<a href="https://github.com/NACLab/ngc-sim-lib">official page</a>)
* NumPy (>=1.22.0)
* SciPy (>=1.7.0)
* JAX (>= 0.4.28; and jaxlib>=0.4.28) <!--(tested for cuda 11.8)-->
* Matplotlib (>=3.8.0), (for `ngclearn.utils.viz`)
* Scikit-learn (>=1.6.1), (for `ngclearn.utils.patch_utils` and `ngclearn.utils.density`)

Note that the above requirements are taken care of if one installs NGC-Learn through either `pip`. One can either install the CPU version of NGC-Learn (if no JAX is pre-installed or only the CPU version of JAX is currently installed) via: 
```console
$ pip install ngclearn
```

or install the GPU version of NGC-Learn by first installing the <a href="https://jax.readthedocs.io/en/latest/installation.html">CUDA 12 version of JAX</a> before running the above pip command.

Alternatively, one may locally, step-by-step (see below), install and setup NGC-Learn from source after pulling from the repo. 

Note that installing the official pip package without any form of JAX installed on your system will default to downloading the CPU version of NGC-Learn (see below for installing the GPU version).

## Install from Source

1. Install NGC-Sim-Lib first (as an editable install); visit the repo https://github.com/NACLab/ngc-sim-lib for details.

2. Clone the NGC-Learn repository: 
```console
$ git clone https://github.com/NACLab/ngc-learn.git
$ cd ngc-learn
```

3. (<i>Optional</i>; only for GPU version) Install JAX for either CUDA 12 , depending    on your system setup. Follow the    <a href="https://jax.readthedocs.io/en/latest/installation.html">installation instructions</a> on the official JAX page to properly install the CUDA 11 or 12 version.

4. Install the NGC-Learn package via:
```console
$ pip install .
```
or, to install as an editable install for development, run:
```console
$ pip install -e .
```

If the installation was successful, you should see the following if you test it against your Python interpreter, i.e., run the <code>$ python</code> command and complete the following sequence of steps as depicted in the screenshot below:<br>

```console
Python 3.11.4 (main, MONTH  DAY YEAR, TIME) [GCC XX.X.X] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import ngclearn
>>> ngclearn.__version__
'3.0.0'
```

<!--
<i>Note</i>: If you do not have a JSON configuration file in place (see tutorials
for details) locally where you call the import to ngc-learn, a warning will pop
up containing within it "<i>UserWarning: Missing file to preload modules from.</i>";
this still means that ngc-learn installed successfully but you will need to
point to a JSON configuration when building projects with ngc-learn.
-->

