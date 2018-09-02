#ifndef DEEPNN
#define DEEPNN
#include"layer.h"

void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], fixo output[10], bit64_t weight1[MAX_W_FC/64], bit64_t weight2[80]);

#endif
