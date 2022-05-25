# cython: c_string_type=str, c_string_encoding=ascii

import os
import numpy as np
cimport numpy as np

from libc.stdint cimport int64_t

np.import_array()


cdef extern from "infimnist.h":
    ctypedef struct infimnist_t

    infimnist_t* infimnist_create(const char*, const float, const int)
    void infimnist_destroy(infimnist_t*)

cdef extern from "py_infimnist.h":
    void get_digits(...)


cdef class InfimnistGenerator:
    cdef infimnist_t* p

    def __init__(self, alpha=1.0, translate=True):
        # specify data path here
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        cdef char* dirname = data_path
        self.p = infimnist_create(dirname, alpha, 1 if translate else 0)

    def __dealloc__(self):
        infimnist_destroy(self.p)

    def gen(self, int64_t[:] indexes not None):
        n_digits = indexes.shape[0]
        digits = np.empty(n_digits * 28 * 28, dtype=np.uint8)
        labels = np.empty(n_digits, dtype=np.uint8)
        cdef char[:] digit_map = digits
        cdef char[:] label_map = labels

        get_digits(&digit_map[0], digit_map.shape[0],
                   &label_map[0], label_map.shape[0],
                   self.p, &indexes[0], indexes.shape[0])

        return digits, labels
