"""
Utility I/O functions.

@author Alexander Ororbia
"""
import io
import sys
import math
import numpy as np
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

def plot_sample_img(x_s, px, py, fname, plt, rotNeg90=False):
    """
    Plots a (1 x (px * py)) array as a (px x py) gray-scale image and
    saves this image to disk.

    Args:
        x_s: the numpy image array

        px: number of pixels in the row dimension

        py: number of pixels in the column dimension

        fname: the filename of the image to save to disk

        plt: a matplotlib plotter object

        rotNeg90: rotates the image -90 degrees before saving to disk
    """
    #px = py = 28
    xs = x_s #x_s.numpy()
    xs = xs.reshape(px, py)
    if rotNeg90 is True:
        xs = np.rot90(xs, -1)
    #xs.resize((px,py))
    my_dpi = 96
    plt.figure(figsize=(800/my_dpi, 1000/my_dpi), dpi=my_dpi)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #             hspace = 0, wspace = 0)
    # plt.margins(0,0)
    plt.imshow(xs, cmap="gray")
    plt.axis("off")
    #plt.tight_layout()
    plt.savefig(fname,dpi=my_dpi, bbox_inches='tight',pad_inches = 0)
    plt.clf()
    plt.close()
