"""
Generic saving/loading utilies.

@author Alexander Ororbia
"""
import io
import sys
import math
import pickle

def serialize(fname, object):
    """
        Serializes an object to disk.

        Args:
            fname: filename of object to save - /path/to/fname_of_model
            model: model object to serialize
    """
    fd = open(fname, 'wb')
    pickle.dump(object, fd)
    fd.close()

def deserialize(fname):
    """
        De-serializes a object from disk

        Args:
            fname: filename of object to load - /path/to/fname_of_model
        Returns:
            the deserialized model object
    """
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object
