#ifndef PY_INFIMNIST_H
#define PY_INFIMNIST_H

#include "infimnist.h"

void get_digits(char* const out_digits, const size_t out_digits_size,
                char* const out_labels, const size_t out_labels_size,
                infimnist_t* p,
                const int64_t* const indexes, const size_t num_digits);

#endif // PY_INFIMNIST_H
