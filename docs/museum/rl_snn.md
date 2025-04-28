# Reinforcement Learning through a Spiking Controller

In this exhibit, we will see how to construct a simple biophysical model for 
reinforcement learning with a spiking neural network and modulated 
spike-timing-dependent plasticity. 
This model incorporates a mechanisms from several different models, including 
the constrained RL-centric SNN of <b>[1]</b> as well as the simplifications 
made with respect to the model of <b>[2]</b>. The model code for this
exhibit can be found
[here](https://github.com/NACLab/ngc-museum/tree/main/exhibits/rl_snn).

## Modeling Operant Conditioning through Modulation

Operant conditioning refers to the idea that there are environmental stimuli that can either increase or decrease the occurrence of (voluntary) behaviors; in other words, positive stimuli can lead to future repeats of a certain behavior whereas negative stimuli can lead to (i.e., punish) a decrease in future occurences. Ultimately, operant conditioning, through consequences, shapes voluntary behavior where actions followed by rewards are repeated and actions followed by punished/negative outcomes diminish. 

In this lesson, we will model very simple case of operant conditioning for a neuronal motor circuit used to engage in the navigation of a simple maze. The maze's design will be the rat T-maze and the "rat" will be allowed to move, at a particular point in the maze, in one of four directions (North, South, West, and East). A positive reward will be supplied to our rat neuronal circuit if it makes progress towards the direction of food (placed in the upper right corner of the T-maze) and a negative reward will be provided if fails to make progress/gets stuck, i.e., a dense reward functional will be employed.



### Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP)


## The Spiking Neural Circuit Model


### Neuronal Dynamics


## Running the RL-SNN Model


<!-- References/Citations -->
## References
<b>[1]</b> Chevtchenko, Sérgio F., and Teresa B. Ludermir. "Learning from sparse 
and delayed rewards with a multilayer spiking neural network." 2020 International 
Joint Conference on Neural Networks (IJCNN). IEEE, 2020. <br> 
<b>[2]</b> Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit
recognition using spike-timing-dependent plasticity." Frontiers in computational
neuroscience 9 (2015): 99.

