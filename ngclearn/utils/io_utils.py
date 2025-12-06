"""
File and OS input/output (reading/writing) utilities.
"""
# import jax
# from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, sys, pickle
from typing import Any

def serialize(fname, object): ## object "saving" routine
    """
    Serialization (saving) routine

    Args:
        fname: file name to serialize to on disk

        object: generic object to serialize
    """
    fd = open(fname, 'wb')
    pickle.dump(object, fd)
    fd.close()

def deserialize(fname): ## object "loading" routine
    """
    Deserialization (loading) routine

    Args:
        fname: file name from disk to deserialize

    Returns:
        deserialized object from disk
    """
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object

def makedir(directory):
    """
    Creates a folder/directory on disk

    Args:
        directory: string name of directory/folder to create on disk
    """
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
    """
    Creates a set of folders/directories on disk

    Args:
        directories: array of string names of directories/folders to create on disk
    """
    for dir in directories:
        makedir(dir)


def save_pkl(directory: str, name: str, value: Any) -> None:
  file_name = directory + "/" + name + ".pkl"
  with open(file_name, 'wb') as f:
    pickle.dump(value, f)

def load_pkl(directory: str, name: str) -> Any:
  file_name = directory + "/" + name + ".pkl"
  with open(file_name, 'rb') as f:
    data = pickle.load(f)
  return data
