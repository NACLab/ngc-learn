<img src="img/ngc-learn-logo.png" width="128">

<b>NGC-Learn-Core</b> - A Toolkit for Building Arbitrary Neural Generative Coding (NGC) models.
This module contains the bare-bones of the NAC-internal library meant to
facilitate research in predictive processing.
This library is meant to internally support CogNGen development (specifically
the NGC-circuits, such as those in the motor/modality cortex models) and not
permitted for public release or distribution.
The public version of this
module, along with its full documentation will be released soon.

To install this module into your system, please follow the basic setup below.

<b>Basic Setup and Dependencies:</b>

Ensure that you have installed the following base dependencies in your system.
Note that this library was developed and tested on Ubuntu 18.04. Install
the following:
1) Have Python 3.x installed on system
2) Have Numpy 1.20.0 installed on system
3) Install Tensorflow 2.0.0, i.e., tensorflow-gpu>=2.0.0

Then run the setup script at the top of the <code>ngclearn_core</code> directory
to have the <code>ngclearn</code> package install on your system using the
following bash command:
<pre>
$ python setup.py install
</pre>

If the installation was successful, you should see the following if you test
it against your Python interpreter, i.e., run the <code>$ python</code> command
and complete the following sequence of steps as depicted in the screenshot below:<br>
<img src="img/test_ngclearn_install.png" width="512">


<b>Attribution:</b>

If you use this code in any form in your project(s), please cite its source
paper:
<pre>
@article{ororbia2020neural,
  title={The neural coding framework for learning generative models},
  author={Ororbia, Alexander and Kifer, Daniel},
  journal={arXiv preprint arXiv:2012.03405},
  year={2020}
}
</pre>

Version:  
0.0.1 Alpha

Author:
Alexander G. Ororbia II
Director, Neural Adaptive Computing (NAC) Laboratory
Rochester Institute of Technology, Department of Computer Science

<b>Copyright:</b>

Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU LGPL-3.0-or-later license.

You should have received a copy of the XYZ license with
this file. If not, please write to: ago@cs.rit.edu , or visit:
https://www.gnu.org/licenses/lgpl-3.0.en.html
