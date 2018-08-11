#include "model_conv.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], bit8_t output[O_WIDTH*O_WIDTH * 64]){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=k1 complete
#pragma HLS ARRAY_PARTITION variable=h1 complete
#pragma HLS ARRAY_PARTITION variable=k2 complete
#pragma HLS ARRAY_PARTITION variable=h2 complete
	bit mem1[64][28][28] = {0};
#pragma HLS ARRAY_PARTITION variable=mem1 complete dim=1
	bit mem2[64][28][28] = {0};
#pragma HLS ARRAY_PARTITION variable=mem2 complete dim=1
	bit32_t mem3[2][14][14];
#pragma HLS ARRAY_PARTITION variable=mem3 complete dim=1


	for (int i = 0; i < I_WIDTH1; i++)
		for (int j = 0; j < I_WIDTH1; j++){
#pragma HLS PIPELINE
			int i_index = i + j*I_WIDTH1;
			mem1[0][j][i] = x[i_index];
		}

	conv_1(mem1, mem2, w_conv1, k1, h1, 1, 32, 32, con1);

	max_pool(mem2, mem3, 32, I_WIDTH1);

	conv_2(mem3[0], mem2, w_conv2, k2, h2, con2);

	max_pool(mem2, mem3, 64, I_WIDTH2);

	for (int m = 0; m < 64; m++)
		for (int i = 0; i < O_WIDTH; i++)
			for (int j = 0; j < O_WIDTH; j++){
#pragma HLS PIPELINE
				int o_index = i + j * O_WIDTH + m * O_WIDTH*O_WIDTH;
				output[o_index] = mem3[m / 32][j][i][m % 32];
			}
}
