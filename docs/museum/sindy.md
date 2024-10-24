# Sparse Identification of Non-linear Dynamical Systems (SINDy)[1]

In this section, we teach, create, simulate, and visualize SINDy model implemented in NGC-Learn library. After going through this demonstration, you will:

1.  Learn how to build a SINDy model of time-series dataset, generated using Ordinary Differential Equations (ODE) of known dynamical systems used in [1].
2.  Learn how to build polynomial libraries of given dataset with arbitrary order.
3.  Learn how to do sparse regression in NGC (Sequential Thresholding Least Square)
4.  Visualize the acquired filters of the learned dictionary models and examine
the results of imposing a kurtotic prior as well as a thresholding function
over latent codes.



The model **code** for this exhibit can be found [here](https://github.com/NACLab/ngc-museum/exhibits/sindy/sindy.py).

## SINDy 
SINDy is a data-driven algorithm that discovers the differential equation governing the dynamical systems. It uses symbolic regression to identify differential equation of the system and it solves sparse regression over the pre-defined library of candidate terms. It takes time series gathered dataset of the system and it gives you its describing differential equation.




<p align="center">
  <img src="../images/museum/sindy/sindy.png" width="900">
</p>


### Inputs
Given a set of time-series measurements of a dynamical system state variables ($\mathbf{X}_{(m \times n)}$) we construct:

Time: $ts = [t_0, t_1, \dots,  T]$  (number of measurements)

State matrix: $\mathbf{X}_{(m \times n)}$  (t measurements of n variables)


### Data Matrix

Given a set of time-series measurements of a dynamical system state variables ($\mathbf{X}_{(m \times n)}$) we construct:
Derivative matrix: $\dot{\mathbf{X}}_{(m \times n)}$ (computed numerically)

Library of Candidate Functions: $\Theta(\mathbf{X}) = [\mathbf{1} \quad \mathbf{X} \quad \mathbf{X}^2 \quad \mathbf{X}^3 \quad \sin(\mathbf{X}) \quad \cos(\mathbf{X}) \quad ...]$

------------------


<p align="center">
  <img src="../images/museum/sindy/lorenz.png" width="300">
  <img src="../images/museum/sindy/oscillator.png" width="300">
</p>

<p align="center">
  <img src="../images/museum/sindy/linear_2D.png" width="300">
  <img src="../images/museum/sindy/cubic_2D.png" width="300">
  <img src="../images/museum/sindy/linear_3D.png" width="300">
</p>






SINDy describes the derivative (linear operation acting on △t) as linear transformations
of a manually constructed dictionary from the state vector by a coefficient matrix.
Dictionary learning combined with LASSO (L1-norm) promotes the sparsity of the coefficient matrix
which allows only governing terms in the dictionary stay non-zero.







