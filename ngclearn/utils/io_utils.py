import io
import sys
import math
import pickle

def serialize(fname, object):
    """
        Serializes object to disk

        fname: /path/to/fname_of_model
        model: model object to serialize
        return: None
    """
    fd = open(fname, 'wb')
    pickle.dump(object, fd)
    fd.close()

def deserialize(fname):
    """
        De-serializes a object from disk

        fname: /path/to/fname_of_model
        return: model object
    """
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object
