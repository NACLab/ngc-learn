#include "infimnist.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


static void write32(FILE* const stream, int x) {
  char buf[4];
  buf[0] = (x>>24)&0xff;
  buf[1] = (x>>16)&0xff;
  buf[2] = (x>>8)&0xff;
  buf[3] = (x>>0)&0xff;
  fwrite(buf,4,1,stream);
} 

void get_digits(char* const out_digits, const size_t out_digits_size,
                char* const out_labels, const size_t out_labels_size,
                infimnist_t* p,
                const int64_t* const indexes, const size_t num_digits) {
  FILE* d_stream = fmemopen((void*)out_digits, out_digits_size + 1, "wb");
  FILE* l_stream = fmemopen((void*)out_labels, out_labels_size + 1, "wb");

  // uncomment for adding headers
  /*
  write32(d_stream, 0x803);
  write32(d_stream, num_digits);
  write32(d_stream, 28);
  write32(d_stream, 28);

  write32(l_stream, 0x801);
  write32(l_stream, num_digits);
  */

  /* data */
  size_t j;
  for (j = 0; j < num_digits; ++j) {
    const long idx = indexes[j];
    // pattern
    const unsigned char *s = infimnist_get_pattern(p, idx);
    fwrite(s,784,1,d_stream);

    // label
    char b = infimnist_get_label(p, idx);
    fwrite(&b,1,1,l_stream);
  }

  fflush(d_stream);
  fflush(l_stream);
}
