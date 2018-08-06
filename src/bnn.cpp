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
	bit mem_conv1[64][32][32] = {0};
#pragma HLS ARRAY_PARTITION variable=mem_conv1 complete dim=1
	bit mem_conv2[64][32][32] = {0};
#pragma HLS ARRAY_PARTITION variable=mem_conv2 complete dim=1
	//fix mem_conv3[64][32][32] = {0};
//#pragma HLS ARRAY_PARTITION variable=mem_conv3 complete dim=1


	for (int i = 0; i < I_WIDTH1; i++)
		for (int j = 0; j < I_WIDTH1; j++){
			int i_index = i + j*I_WIDTH1;
			mem_conv2[0][j][i] = x[i_index];
		}

	pad(mem_conv2, mem_conv1, 1, I_WIDTH1);

	conv_2d(mem_conv1, mem_conv2, w_conv1, k1, h1, 1, 32, 32, con1);

	//batch_norm(mem_conv3, mem_conv1, k1, h1, 32, I_WIDTH1);

	max_pool(mem_conv2, mem_conv1, 32, I_WIDTH1);

	pad(mem_conv1, mem_conv2, 32, I_WIDTH2);

	conv_2d(mem_conv2, mem_conv1, w_conv2, k2, h2, 32, 64, 18, con2);

	//batch_norm(mem_conv3, mem_conv1, k2, h2, 64, I_WIDTH2);

	max_pool(mem_conv1, mem_conv2, 64, I_WIDTH2);

	for (int m = 0; m < 64; m++)
		for (int i = 0; i < O_WIDTH; i++)
			for (int j = 0; j < O_WIDTH; j++){
				int o_index = i + j * O_WIDTH + m * O_WIDTH*O_WIDTH;
				output[o_index] = mem_conv2[m][j][i];
			}
}
