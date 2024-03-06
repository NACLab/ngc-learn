"""
File and OS input/output (reading/writing) utilities.
"""
import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, sys, pickle

def serialize(fname, object): ## object "saving" routine
    fd = open(fname, 'wb')
    pickle.dump(object, fd)
    fd.close()

def deserialize(fname): ## object "loading" routine
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object


def makedir(directory):
    folders = directory.split('/')
    prefix = ""
    remove = []
    for idx, f in enumerate(folders):
        if f == ".." or f == ".":
            prefix = prefix + f + "/"
            remove.append(idx)

    for i in reversed(remove):
        folders.pop(i)

    for folder in folders:
        if not os.path.isdir(prefix + folder):
            os.mkdir(prefix + folder)
        prefix = prefix + folder + "/"

def makedirs(directories):
    for dir in directories:
        makedir(dir)
