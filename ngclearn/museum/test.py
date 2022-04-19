import os
import sys, getopt, optparse
import pickle
import tensorflow as tf
import numpy as np
sys.path.insert(0, '../../')
from ngclearn.museum.gncn_t2_ffn import GNCN_t2_FFN

# create dummy data
x = tf.expand_dims(tf.cast([0.25,-0.13,0.345,-0.2, 0.5],dtype=tf.float32),axis=0)
t = tf.expand_dims(tf.cast([0.6,-0.4,0.7,-0.31],dtype=tf.float32),axis=0)

n_s = 3
x = tf.random.uniform(shape=(n_s,5), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=69)
t = tf.random.uniform(shape=(n_s,3), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=69)

in_dim = x.shape[1]
out_dim = t.shape[1]

n_iter = 50 #100
#n_iter = 200

has_bias = True
project_init = True # <-- NOTE: speeds up settling cycle convergence greatly
act_fx = "relu"
K = 10 # 20 #15 # 10 #15 # 30 #20
beta = 0.1 #0.01 #0.05
n_hid = 100 #200 #100 #64 #32 # 128 #64 # 32
wght_sd = 0.1 #0.025 #0.1 # 0.055
leak = 0.0001 #0.05 #0.0001
agent = GNCN_t2_FFN(in_dim, out_dim, z_dim=n_hid, act_fx=act_fx, out_fx="identity",
                    K=K, beta=beta, leak=leak, seed=69, wght_sd=wght_sd, has_bias=has_bias)

# K=10
# t_hat = agent.settle(input=x, target=None, project_init=project_init, debug=True)
#
# sys.exit(0)

for iter in range(n_iter):
    print("###################################################################")
    print("Iter({})".format(iter))
    #print("Setting:")
    # agent.ngc_model.inject("e2", ("avg_scalar", x.shape[0] * 1.0))
    # agent.ngc_model.inject("e1", ("avg_scalar", x.shape[0] * 1.0))
    # agent.ngc_model.inject("e0", ("avg_scalar", x.shape[0] * 1.0))
    t_hat = agent.settle(input=x, target=t, project_init=project_init)
    #print("Input   - x    : {}".format(agent.ngc_model.extract("z3","phi(z)").numpy()))
    print("-------------------------------")
    e2 = agent.ngc_model.extract("e2","phi(z)")
    e1 = agent.ngc_model.extract("e1","phi(z)")
    e0 = agent.ngc_model.extract("e0","phi(z)")
    L_l = tf.reduce_sum(e2 * e2) #* x.shape[0])
    print("L2  : {}".format(L_l))
    L_l = tf.reduce_sum(e1 * e1) #* x.shape[0])
    print("L1  : {}".format(L_l))
    L_l = tf.reduce_sum(e0 * e0) #* x.shape[0])
    print("L0  : {}".format(L_l))
    # print("e2  : {}".format(tf.norm(e2,ord=2)))
    # print("e1  : {}".format(tf.norm(e1,ord=2)))
    # print("e0  : {}".format(tf.norm(e0,ord=2)))
    print("-------------------------------")

    #print("Settle  - t.hat: \n{}".format(t_hat.numpy()))
    agent.update(x, avg_update=False)
    agent.evolve_err_weights()
    agent.clear()

    # ------------ test model query functions ---------------

    t_hat = agent.settle(input=x, target=t, project_init=project_init)
    print("z0:\n",agent.ngc_model.extract("z0","phi(z)"))
    print("Settle  - t.hat: \n{}".format(t_hat.numpy()))
    agent.clear()

    # Projection/ancestral sampling to get fast estimate of p(y|x) (improper)
    t_hat = agent.predict(input=x)
    print("Project - t.hat: \n{}".format(t_hat.numpy()))
    agent.clear()

    # Free-form settling (unclamped output/target nodes), infer p(y|x)
    t_hat = agent.settle(input=x, target=None, project_init=True)

    print("z0:\n",agent.ngc_model.extract("z0","phi(z)"))
    #print("t    :\n{}".format(t.numpy()))
    #print("Input   - x    : {}".format(agent.ngc_sampler.extract("s3","phi(z)").numpy()))
    print("I-Infer - t.hat: \n{}".format(t_hat.numpy()))
    agent.clear()
    print("Target  - t    : \n{}".format(t.numpy()))
