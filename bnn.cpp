#include "layer.h"
#include "model_conv.h"
#include "typedef.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

void bnn(bit x[I_WIDTH1 * I_WIDTH1], bit output[O_WIDTH*O_WIDTH * 64]){
	bit mem_conv1[64][32][32] = {0};
#pragma HLS ARRAY_PARTITION variable=mem_conv1 block factor=64
	bit mem_conv2[64][32][32] = {0};
#pragma HLS ARRAY_PARTITION variable=mem_conv2 block factor=64
	fix mem_conv3[64][32][32] = { 0 };
#pragma HLS ARRAY_PARTITION variable=mem_conv3 block factor=64


	for (int i = 0; i < I_WIDTH1; i++)
		for (int j = 0; j < I_WIDTH1; j++){
			int i_index = i + j*I_WIDTH1;
			mem_conv2[0][j][i] = x[i_index];
		}

	pad(mem_conv2, mem_conv1, 1, I_WIDTH1);

	conv_2d(mem_conv1, mem_conv3, w_conv1, 1, 32, 32);

	batch_norm(mem_conv3, mem_conv1, miu1, sigma1, gamma1, beta1, 32, I_WIDTH1);

	max_pool(mem_conv1, mem_conv2, 32, I_WIDTH1);

	pad(mem_conv2, mem_conv1, 32, I_WIDTH2);

	conv_2d(mem_conv1, mem_conv3, w_conv2, 32, 64, 18);

	batch_norm(mem_conv3, mem_conv1, miu2, sigma2, gamma2, beta2, 64, I_WIDTH2);

	max_pool(mem_conv1, mem_conv2, 64, I_WIDTH2);

	for (int m = 0; m < 64; m++)
		for (int i = 0; i < O_WIDTH; i++)
			for (int j = 0; j < O_WIDTH; j++){
				int o_index = i + j * O_WIDTH + m * O_WIDTH*O_WIDTH;
				output[o_index] = mem_conv2[m][j][i];
			}
}
