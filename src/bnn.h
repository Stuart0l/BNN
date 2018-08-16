#ifndef DEEPNN
#define DEEPNN
#include"layer.h"

//#pragma SDS data zero_copy(x[0:784], output[0:3136])
//#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS, output:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(x:SEQUENTIAL, output:SEQUENTIAL)
void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], bit8_t output[O_WIDTH*O_WIDTH * 64]);

#endif
