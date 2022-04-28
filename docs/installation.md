# Installation

**ngc-learn** officially supports Linux on Python 3. It can be run with or
without a GPU.

<i>Setup:</i> Ensure that you have installed the following base dependencies in
your system. Note that this library was developed and tested on Ubuntu 18.04.
Specifically, ngc-learn requires:
* Python (>=3.7)
* Numpy (>=1.20.0)
* Tensorflow 2.0.0, specifically, tensorflow-gpu>=2.0.0
* scikit-learn (>=0.24.2) if using ngclearn.density (needed for the examples/)

You can install the dependencies above (and a few extras needed to
construct the docs) by running:

```console
$ pip install -r requirements.txt
```

## Install from Source

1. Clone the ngc-learn repository:
```console
$ git clone https://github.com/ago109/ngc-learn.git
$ cd ngc-learn
```

2. Install the base requirements with:
```console
$ pip3 install -r requirements.txt
```

3. Install the ngc-learn package via:
```console
$ python setup.py install
```

If the installation was successful, you should see the following if you test
it against your Python interpreter, i.e., run the <code>$ python</code> command
and complete the following sequence of steps as depicted in the screenshot below:<br>
<img src="images/test_ngclearn_install.png" width="512">
