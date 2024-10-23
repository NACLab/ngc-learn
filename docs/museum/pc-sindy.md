# Sparse Identification of Non-linear Dynamical Systems (SINDy)[]

In this section, we teach, create, simulate, and visualize SINDy model implemented in NGC-Learn library. 







The model **code** for this exhibit can be found [here](https://github.com/NACLab/ngc-learn/sindy/sindy.py).

## SINDy 
SINDy is a data-driven algorithm that discovers the differential equation governing the dynamical systems. It ...


in terms of a symbolic equation of the system's state vector. 
SINDy describes the derivative (linear operation acting on △t) as linear transformations
of a manually constructed dictionary from the state vector by a coefficient matrix.
Dictionary learning combined with LASSO (L1-norm) promotes the sparsity of the coefficient matrix
which allows only governing terms in the dictionary stay non-zero.

Learning the 


## SINDy with Predictive Coding 

<p align="center">
  <img src="" width="550" title="">
</p>

## Predictive Coding Model Dynamics
