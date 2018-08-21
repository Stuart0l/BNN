#ifndef DEEPNN
#define DEEPNN
#include"layer.h"

//#pragma SDS data zero_copy(x[0:784], output[0:3136])
//#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS, output:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(x:SEQUENTIAL, output:SEQUENTIAL)
#pragma SDS data mem_attribute(weight1:PHYSICAL_CONTIGUOUS, weight2:PHYSICAL_CONTIGUOUS)
#pragma SDS data zero_copy(weight1, weight2)
void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], fixo output[10], bit8_t weight1[MAX_W_FC/8], bit8_t weight2[640]);

#endif
