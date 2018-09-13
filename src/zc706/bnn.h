#ifndef DEEPNN
#define DEEPNN
#include"layer.h"

//#pragma SDS data zero_copy(x[0:784], output[0:3136])
//#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS, output:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(x:SEQUENTIAL, output:SEQUENTIAL, weight1:SEQUENTIAL, weight2:SEQUENTIAL)
#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS, weight1:PHYSICAL_CONTIGUOUS, weight2:PHYSICAL_CONTIGUOUS)
void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], fixo output[10], bit64_t weight1[MAX_W_FC/64], bit64_t weight2[80]);

#endif
