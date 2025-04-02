<!--
Make a working code
Results section jumps to complex examples without first showing a simple case
No clear connection between the code section and the theoretical explanation
Missing explanation of hyperparameter selection (threshold, max_iter, etc.)
Some diagrams (like P1.png and P2.png) are too small to read clearly
Flow diagrams lack clear directional indicators
Inconsistent color schemes across visualizations
-->


# Sparse Identification of Non-linear Dynamical Systems (SINDy)[1]

In this section, we will study, create, simulate, and visualize a model known as the sparse identification of non-linear dynamical systems (SINDy) [1], implementing it in NGC-Learn and JAX. After going through this demonstration, you will:

1.  Learn how to uncover the differential equations of a dynamical system using the SINDy algorithm, using only snapshots of the system of interest.
2.  Learn how to build polynomial libraries of an arbitrary order from a dataset.
3.  Learn how to solve the sparse regression problem in two ways: 
  - Iteratively finding the coefficient matrix by gradient descent.
  - Iteratively performing the least squares (LSQ) method followed by thresholding, i.e., carrying out sequential thresholding least squares (STLSQ) for the given model.
   
   
The model **code** for this exhibit (in NGC-Museum) can be found [here](https://github.com/NACLab/ngc-museum/exhibits/sindy/sindy.py).

## The SINDy Process Model 

SINDy is a data-driven algorithm that discovers the governing behavior of a dynamical system in terms of symbolic differential equations. It solves the sparse regression problem over the coefficients of a pre-defined library that includes $p$ candidate predictors. It ultimately tries to find sparse model that only uses $s$ predictors out of $p$ where $s<p$ that best describes the dynamics (time-derivatives) of the system only from a dataset collected over time. SINDy assumes that the systems it is to identify follow the parsimonious theorem (Occam's Razor) where the balance between the complexity and accuracy results in effective generalization.

### SINDy Dynamics

If $\mathbf{X}$ is a system that only depends on variable $t$, a very small change in the independent variable ($dt$) can cause a change in the system by $dX$ amount. 
$$$
d\mathbf{X} = \mathbf{Ẋ}(t)~dt
$$$
SINDy models the derivative[^1] (a linear operation) as linear transformations with:
[^1]: The derivative is a linear operation that acts on $dt$ and gives a differential that is the linearized approximation of the taylor series of the function.
$$$
\frac{d\mathbf{X}(t)}{dt} = \mathbf{Ẋ}(t) = \mathbf{f}(\mathbf{X}(t))
$$$
SINDy assumes that this linear operation, i.e., $\mathbf{f}(\mathbf{X}(t))$, is a matrix multiplication that linearly combines the relevant predictors in order to describe the system's equation.
$$$
\mathbf{f}(\mathbf{X}(t)) = \mathbf{\Theta}(\mathbf{X})~\mathbf{W}
$$$

Given a group of candidate functions within the library $\mathbf{\Theta}(\mathbf{X})$, the coefficients in $\mathbf{W}$ that choose the library terms are to be **sparse**. In other words, there are only a few functions that exist in the system's differential equation. Given these assumptions, SINDy solves a sparse regression problem in order to find the $\mathbf{W}$ that maps the library of selected terms to each feature of the system being identified. SINDy imposes parsimony constraints over the resulting symbolic regression (i.e., genetic programming) to describe a dynamical system's behavior with as few terms as possible. In order to select a sparse set of the given features, the model adds the LASSO regularizarion penalty (i.e., an L1 norm constraint) to the regression problem and solves the sparse regression or solves the regression problem via STLSQ. We will describe STLSQ in third step of the SINDy dynamics/process.

In essence, SINDy's dynamics can be presented in three main phases, visualized in Figure 1 below. 

------------------------------------------------------------------------------------------

<p align="center">
  <img src="../images/museum/sindy/flow_SR.png" width="500">


**Figure 1:** **The flow of the three phases in SINDy.** **Phase-1)** Data collection: capturing system states that are changing in time and creating the state vector. **Phase-2A)** Library formation: manually creating the library of candidate predictors that could appear in the model. **Phase-2B)** Derivative computation: using the data collected in phase 1 to compute its derivative with respect to time. **Phase-3)**  Solving the sparse regression problem.
</p>

------------------------------------------------------------------------------------------

<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

<table>
   
<tr>
<td width="70%" valign="top">
   
## Phase 1: Collecting Dataset → $\mathbf{X}_{(m \times n)}$
This phase involves gathering the raw data points representing the system's states across time. In this example, this means capturing the $x$, $y$, and $z$ coordinates of the system's states. Here, $m$ represents the number of data points (number of the snapshots/length of time) and $n$ is the system's dimensionality.
</td>
<td width="30%" align="top">
   <p align="center">
   <img src="../images/museum/sindy/P1.png" width="100" alt="Dataset collection showing x, y, z coordinates">
   <img src="../images/museum/sindy/X_.png" width="100">
   </p>
</td>
</tr>

</table>
<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->
<table>


   
<tr>
   <td colspan="2"> 
     
## Phase 2: Processing
   </td>
     <td rowspan="3" colspan="5"> 
   <p align="center">
   <img src="../images/museum/sindy/P2.png" width="10000">
   </p>
   </td>
</tr>

   
<tr>
   <td> 

### 2.A: Making the Library  → $\mathbf{\Theta}_{(m \times p)}$
In this step, using the dataset collected in phase 1, given pre-defined function terms, we construct a dictionary of candidate predictors for identifying the target system's differential equations. These functions form the columns of our library matrix $\mathbf{\Theta}(\mathbf{X})$ and $p$ is the number of candidate predictors. To identify the dynamical structure of the system, this library of candidate functions appears in the regression problem to propose the model's structure that will later serve as the coefficient matrix for weighting the functions according to the problem setup. We assume sparse models will be sufficient to identify the system and do this through sparsification (LASSO or thresholding weights) in order decide which structure best describes the system's behavior using predictors. 
Given a set of time-series measurements of a dynamical system state variables ($\mathbf{X}_{(m \times n)}$) we construct the following:
Library of Candidate Functions: $\Theta(\mathbf{X}) = [\mathbf{1} \quad \mathbf{X} \quad \mathbf{X}^2 \quad \mathbf{X}^3 \quad \sin(\mathbf{X}) \quad \cos(\mathbf{X}) \quad ...]$
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/Xtheta.png" width="3000">
   </p>
   </td>
</tr>


<tr>
   <td> 
   
### 2.B: Compute State Derivatives → $\mathbf{Ẋ}_{(m \times n)}$
Given a set of time-series measurements of a dynamical system's state variables $\mathbf{X}_{(m \times n)}$, we next construct the derivative matrix: $\mathbf{Ẋ}_{(m \times n)}$ (computed numerically). In this step, using the dataset collected in phase 1, we compute the derivatives of each state variable with respect to time. In this example, we compute $ẋ$, $ẏ$, and $ż$ in order to capture how the system evolves over time.
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/xdx.png" width="200">
   </p>
   </td>
</tr>

<!-- <img src="../images/museum/sindy/dX_.png" width="200"> -->



</table>
<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

<table>
   
<tr>
<td width="70%" valign="top">
   
## Phase 3: Solving Sparse Regression Problem → $\mathbf{W_s}_{(p \times n)}$
Solving the resulting sparse regression (SR) problem that results from the phases/steps above can be done using various method such as Lasso, STLSQ, Elastic Net, as well as many other schemes. Here, we describe the STLSQ approach to solve the SR problem according to the SINDy process.
</td>

<td width="30%" align="top">
   <p align="center">
   <img src="../images/museum/sindy/SRin.png" width="390" alt="Dataset collection showing x, y, z coordincates">
   </p>
</td>


</tr>

<tr>
   <table>  
   <tr>
     <td colspan="3"> 



### Solving Sparse Regression by Sequential Thresholding Least Squares (STLSQ)
<!-- --------------------------------------------------------------------------------------------- -->
<p align="center">
  <img src="../images/museum/sindy/flow.png" width="800">

**Figure 1:** **The flow of three phases in SINDy.** **Phase-1)** Data collection: capturing system's states that are changing in time and making the state vector. **Phase-2A)** Library formation: manually creating the library of candidate predictors that could appear in the model. **Phase-2B)** Derivative computation: using the data collected in phase 1 and computing its derivative with respect to time. **Phase-3)**  Solving the sparse regression problem via STLSQ.
</p>

------------------------------------------------------------------------------------------
   </td>
</tr>  


   <tr>
     <td colspan="3"> 
   
### Sequential Thresholding Least Square (STLSQ)
   </td>
</tr>  


   <tr>
     <td colspan="3"> 
        <p align="center">
   <img src="../images/museum/sindy/STLSQ.png" width="800" alt="State derivatives visualization">
   </p>
   </td>
</tr>  


<tr>
   <td> 

#### 3.A: Least Square method (LSQ) → $\mathbf{W}$ 
This step entails finding library coefficients by solving the following regression problem $\mathbf{Ẋ} = \mathbf{\Theta}\mathbf{W}$ analytically $\mathbf{W}  = (\mathbf{\Theta}^T \mathbf{\Theta})^{-1} \mathbf{\Theta}^T \mathbf{Ẋ}$ 
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/LSQ.png" width="200" alt="State derivatives visualization">
   </p>
   </td>
</tr>

<tr>
   <td> 
   
#### 3.B: Thresholding → $\mathbf{W_s}$
This step entails sparsifying $\mathbf{W}$ by keeping only some of the terms within $\mathbf{W}$, particularly those that correspond to the effective terms in the library.
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/Thresholding.png" width="200" alt="State derivatives visualization">
   </p>
   </td>
</tr>
<tr>
   <td> 
   
#### 3.C: Masking → $\mathbf{\Theta_s}$
This step sparsifies $\mathbf{\Theta}$ by keeping only the corresponding terms in $\mathbf{W}$ that remain (from the prior step).
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/Masking.png" width="200" alt="State derivatives visualization">
   </p>
   </td>
</tr>


<tr>
   <td> 
   
#### 3.D: Repeat A → B → C until convergence
We continue to solve LSQ with the sparse matrix $\mathbf{\Theta_s}$ and $\mathbf{W_s}$ and find a new $\mathbf{W}$, repeating steps B and C until convergence.
   </td>
   <td> 
   <p align="center">
   <img src="../images/museum/sindy/iterin.png" width="500" alt="State derivatives visualization">
   </p>
   </td>
</tr>


</table>
</tr>


</table>


<!--
   <p align="center">
   <img src="../images/museum/sindy/dx.png" width="300">
   <img src="../images/museum/sindy/dy.png" width="300">
   <img src="../images/museum/sindy/dz.png" width="300">
   </p>
   -->



<!-- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -->
<!-- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -->
## Code: Simulating SINDy

We finally present ngc-learn code below for using and simulating the SINDy process to identify several dynamical systems.

<!--
-->

```python



import numpy as np
import jax.numpy as jnp
from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
from ngclearn.utils.diffeq.ode_utils import solve_ode
from ngclearn.utils.diffeq.odes import lorenz, linear_2D

jnp.set_printoptions(suppress=True, precision=5)


## system's ode function
dfx = lorenz

x0 = jnp.array([-8, 7, 27], dtype=jnp.float32)    ## initial state

t0 = 0.                             ## starting time
dt = 1e-2                           ## time steps
T = 2000                            ## #of steps

deg = 2                       ## polynomial library degree
include_bias = False          ## if include bias in making poly library
threshold = 0.02              ## sparaity threshold
max_iter=10          

## Phase 1: Collecting Dataset (solving ode)
ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)

## Phase 2.A: Making Library (polynomial library) 
lib_creator = PolynomialLibrary(poly_order=deg, include_bias=include_bias)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

## Phase 2.B: Compute State Derivatives
dX = jnp.array(np.gradient(X, ts.ravel(), axis=0))

##########  Solving Sparse Regression (for each dimension) ##########
#~~~~~~~~~~~~  By Seqyential Thresholding Least Square  ~~~~~~~~~~~~~

for dim in range(dX.shape[1]):
    ## 3.A: 'Initial' Least Square
    coef = jnp.linalg.lstsq(feature_lib, dX[:, dim][:, None], rcond=None)[0]
    
    for i in range(max_iter):
        coef_pre = jnp.array(coef)
        coef_zero = jnp.zeros_like(coef)
        
        ## 3.B: thresholding
        res_idx = jnp.where(jnp.abs(coef) >= threshold,
                                              True,
                                              False)
        ## 3.C: masking
        res_mask = jnp.any(res_idx, axis=1)     ## residual mask
        res_lib = feature_lib[:, res_mask]      ## residual predictors

        ## 3.A: Least Square
        coef_new = jnp.linalg.lstsq(res_lib, dX[:, dim][:, None],
                                    rcond=None
                                    )[0]        ## least square
        
        coef = coef_zero.at[res_mask].set(coef_new)
        
    ## 3.B: 'Final' thresholding
    coeff = jnp.where(jnp.abs(coef) >= threshold, coef, 0.)

    print(f"coefficients for dimension {dim+1}: \n", coeff.T)



```


<!-- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -->
<!-- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -->

## Results: System Identification

Running the above code should produce results similar to the findings we present next.

<table>
<th>
   Model
</th>
<th>
   Results
</th>

<tr>
   <td> 
   
   ## Oscillator

True model's equation \
$\mathbf{ẋ} = \mu_1\mathbf{x} + \sigma \mathbf{xy}$ \
$\mathbf{ẏ} = \mu_2\mathbf{y} + (\omega + \alpha \mathbf{y} + \beta \mathbf{z})\mathbf{z} - \sigma \mathbf{x}^2$ \
$\mathbf{ż} = \mu_2\mathbf{z} - (\omega + \alpha \mathbf{y} + \beta \mathbf{z})\mathbf{y}$

```python
--- SINDy results ----
ẋ =  0.050 𝑥 + 1.100 𝑥𝑦 
ẏ =  2.999 𝓏 -4.999 𝓏^2
     -0.010 𝑦 -1.998 𝑦𝓏 -1.100 𝑥^2 
ż = -0.010 𝓏 -3.000 𝑦
   + 5.000 𝑦𝓏 + 1.999 𝑦^2

  [1,  𝓏, 𝓏^2, 𝑦,  𝑦𝓏, 𝑦^2, 𝑥,     𝑥𝓏,  𝑥𝑦,   𝑥^2]
[[ 0.  0.  0.  0.  0.  0.  0.049  0.  1.099  0.]
 [ 0.  2.99 -4.99 -0.010 -1.99  0.  0.  0.  0. -1.099]
 [ 0. -0.009  0. -2.99  4.99  1.99  0.  0.  0.  0.]]
```

   </td>
   <td> 
     <p align="center">
      <img src="../images/museum/sindy/oscillator.png" width="250">
      <img src="../images/museum/sindy/O3D.png" width="250">
     </p>
   </td>
</tr>

<tr>
   <td> 
   
   ## Lorenz
   
True model's equation \
$\mathbf{ẋ} = 10(\mathbf{y} - \mathbf{x})$ \
$\mathbf{ẏ} = \mathbf{x}(28 - \mathbf{z}) - \mathbf{y}$ \
$\mathbf{ż} = \mathbf{xy} - \frac{8}{3}~\mathbf{z}$



```python
--- SINDy results ----
ẋ =  9.969 𝑦 -9.966 𝑥 
ẏ = -0.972 𝑦 + 27.833 𝑥 -0.995 𝑥𝓏 
ż = -2.657 𝓏 + 0.997 𝑥𝑦

  [𝓏, 𝓏^2,  𝑦,    𝑦𝓏, 𝑦^2, 𝑥,     𝑥𝓏, 𝑥𝑦, 𝑥^2]
[[ 0.  0.  9.968  0.  0. -9.965  0.  0.  0.]
 [ 0.  0. -0.971  0.  0.  27.832 -0.995  0.  0.]
 [-2.656  0.  0.  0.  0.  0.  0.  0.996  0.]]
```

   </td>
   <td> 
     <p align="center">
      <img src="../images/museum/sindy/lorenz.png" width="250">
      <img src="../images/museum/sindy/Lorenz3D.png" width="250">
     </p>
   </td>
</tr>

<tr>
   <td> 
   
   ## Linear-2D

True model's equation \
$\mathbf{ẋ} = -0.1\mathbf{x} + 2.0\mathbf{y}$ \
$\mathbf{ẏ} = -2.0\mathbf{x} - 0.1\mathbf{y}$ 

```python
--- SINDy results ----
ẋ =  2.000  𝑦 -0.100  𝑥 
ẏ = -0.100  𝑦 -2.000  𝑥

[𝑦, 𝑦^2, 𝑥, 𝑥𝑦, 𝑥^2]
[[ 1.999  0. -0.100  0.  0.]
 [-0.099  0. -1.999  0.  0.]]
```
   

   </td>
   <td> 
     <p align="center">
      <img src="../images/museum/sindy/linear_2D.png" width="250">
      <img src="../images/museum/sindy/L2D.png" width="250">
     </p>
   </td>
</tr>

<tr>
   <td> 
   
   ## Linear-3D

True model's equation \
$\mathbf{ẋ} = -0.1\mathbf{x} + 2\mathbf{y}$ \
$\mathbf{ẏ} = -2\mathbf{x} - 0.1\mathbf{y}$ \
$\mathbf{ż} = -0.3\mathbf{z}$ 

```python
--- SINDy results ----
ẋ =  2.000 𝑦 -0.100 𝑥 
ẏ = -0.100 𝑦 -2.000 𝑥 
ż = -0.300 𝓏

[1, 𝓏, 𝓏^2, 𝑦, 𝑦.𝓏, 𝑦^2, 𝑥, 𝑥𝓏, 𝑥.𝑦, 𝑥^2]
[[ 0.  0.  1.999  0.  0. -0.100  0.  0.  0.]
 [ 0.  0. -0.100  0.  0. -1.999  0.  0.  0.]
 [-0.299  0.  0.  0.  0.  0.  0.  0.  0.]]
```

   </td>
   <td> 
     <p align="center">
      <img src="../images/museum/sindy/linear_3D.png" width="250">
      <img src="../images/museum/sindy/L3D.png" width="250">
     </p>
   </td>
</tr>




<tr>
   <td> 
   
   ## Cubic-2D

True model's equation \
$\mathbf{ẋ} = -0.1\mathbf{x}^3 + 2.0\mathbf{y}^3$ \
$\mathbf{ẏ} = -2.0\mathbf{x}^3 - 0.1\mathbf{y}^3$ 

```python
--- SINDy results ----
ẋ =  1.999  𝑦^3 -0.100  𝑥^3 
ẏ = -0.099  𝑦^3 -1.998  𝑥^3

[𝑦, 𝑦^2, 𝑦^3, 𝑥, 𝑥𝑦, 𝑥𝑦^2, 𝑥^2, 𝑥^2𝑦, 𝑥^3]
[[ 0.  0.  1.99  0.   0.   0.   0.   0. -0.100]
 [ 0.  0. -0.099  0.   0.   0.   0.   0. -1.99]]
```

   </td>
   <td> 
     <p align="center">
      <img src="../images/museum/sindy/cubic_2D.png" width="250">
      <img src="../images/museum/sindy/C2D.png" width="250">
     </p>
   </td>
</tr>
   
</table>


## References
<b>[1]</b> Brunton SL, Proctor JL, Kutz JN. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proc Natl Acad Sci U S A. 2016 Apr 12;113(15):3932-7. doi: 10.1073/pnas.1517384113. Epub 2016 Mar 28. PMID: 27035946; PMCID: PMC4839439.

