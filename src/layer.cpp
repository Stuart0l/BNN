#include "layer.h"
#include <cmath>
#include <iostream>
using namespace std;

inline bool if_mac(int x, int y, int I)
{
	if (x < PADDING / 2 || x >= (I - PADDING / 2) || y < PADDING / 2 || y >= (I - PADDING / 2))
		return false;
	return true;
}

inline void load_weight(int n, int m, int N, bit w_buff[128][5][5], const bit w[MAX_W_CONV])
{
	for (int wm = 0; wm < 8; wm++)
	{
		for (int wn = 0; wn < 16; wn++)
		{
			for (int i = 0; i < F; i++)
			{
				for (int j = 0; j < F; j++)
				{
#pragma HLS PIPELINE rewind				
					int w_index = i + j * F + (n + wn + (m + wm) * N) * FILTER_SIZE;
					w_buff[wm*16+wn][j][i] = w[w_index];
				}
			}
		}
	}
}

inline void load_ifmap(int m, bit i_buff[8][28][28], bit in[64][28][28])
{

	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
#pragma HLS PIPELINE rewind		
			for (int fm = 0; fm < 8; fm++)
			{
				i_buff[fm][j][i] = in[m + fm][j][i];
			}
		}
	}
}

inline void store_ofmap(int n, fix o_buff[16][28][28], bit out[64][28][28], const fix k[MAX_F], const fix h[MAX_F])
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
#pragma HLS PIPELINE
			for (int fn = 0; fn < 16; fn++)
			{
#pragma HLS UNROLL
				bit tmp = (o_buff[fn][j][i]*k[n + fn]+h[n + fn]).is_neg() ? 0 : 1;
				out[n + fn][j][i] = tmp;
				o_buff[fn][j][i] = 0.;
			}
		}
	}
}

// @param[in] : input - input fmaps
//              weight - filters
//              M - number of input fmaps
//              N - number of output fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
void conv_1(bit input[64][28][28], bit output[64][28][28], const bit weight[MAX_W_CONV], const fix k[MAX_F], const fix h[MAX_F], int M, int N, int I, fix con)
{
	bit input_buffer[8][28][28];
#pragma HLS ARRAY_PARTITION variable = input_buffer complete dim = 1
	fix output_buffer[16][28][28];
#pragma HLS ARRAY_PARTITION variable = output_buffer complete dim = 1
	fix calc_buffer[16];
#pragma HLS ARRAY_PARTITION variable = calc_buffer complete
	bit weight_buffer[128][5][5];
#pragma HLS ARRAY_PARTITION variable = weight_buffer complete dim = 1
	int O = I - F + 1;

	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			for (int n = 0; n < 16; n++)
#pragma HLS UNROLL
				output_buffer[n][i][j] = 0;

	for (int n = 0; n < 16; n++)
#pragma HLS UNROLL
		calc_buffer[n] = 0;

n:
	for (int n = 0; n < N; n += 16){
	m:
		for (int m = 0; m < M; m += 8){
			load_weight(n, m, N, weight_buffer, weight);
			load_ifmap(m, input_buffer, input);
		x:
			for (int x = 0; x < O; x++){
			y:
				for (int y = 0; y < O; y++){
				c:
					for (int c = 0; c < F; c++){
					r:
						for (int r = 0; r < F; r++){
#pragma HLS PIPELINE rewind
							if (if_mac(x + c, y + r, I)){
								for (int nn = 0; nn < 16; nn++){
									fix tmp_out = 0;
									if (nn + n < N){
										for (int mm = 0; mm < 8; mm++){
											if (mm + m < M){
												tmp_out += ((input_buffer[mm][y + r - PADDING / 2][x + c - PADDING / 2] == weight_buffer[mm * 16 + nn][r][c]) ? con : con.getNeg()); //do not use -, use getNeg
											}
										}
									}
									calc_buffer[nn] += tmp_out;
								}
							}
						}
					}
					for (int nc = 0; nc < 16; nc++){
#pragma HLS UNROLL
						output_buffer[nc][y][x] += calc_buffer[nc];
						calc_buffer[nc] = 0;
					}
				}
			}
		}
		store_ofmap(n, output_buffer, output, k, h);
	}
}

void max_pool(bit input[64][28][28], bit32_t output[2][14][14], int M, int I){
	int O = I / 2;

	for (int j = 0; j < 14; j++)
		for (int i = 0; i < 14; i++)
			for (int m = 0; m < 2; m++)
#pragma HLS UNROLL
				output[m][j][i] = 0;

	for (int x = 0; x < 14; x++){
		if (x < O){
			for (int y = 0; y < 14; y++){
				if (y < O){
					for (int m = 0; m < 64; m++){
#pragma HLS UNROLL factor=2
						if (m < M){
							bit max = 0;
							for (int c = 0; c < 2; c++){
								for (int r = 0; r < 2; r++){
#pragma HLS PIPELINE rewind
									bit tmp;
									if ((tmp = input[m][2 * y + r][2 * x + c])){
										max = tmp;
									}
								}
							}
							output[m / 32][y][x][m % 32] = max;
						}
					}
				}
			}
		}
	}
}

void conv_2(bit32_t input[14][14], bit output[64][28][28], const bit weight[MAX_W_CONV], const fix k[MAX_F], const fix h[MAX_F], fix con){
	bit32_t w_buff[64][5][5];
#pragma HLS ARRAY_PARTITION variable=w_buff complete dim=1
	int count[64];
#pragma HLS ARRAY_PARTITON variable=count complete
	for (int n = 0; n < 64; n++)
#pragma HLS UNROLL
		count[n] = 0;

	for (int m = 0; m < 32; m++){
		for (int c = 0; c < F; c++){
			for (int r = 0; r < F; r++){
#pragma HLS PIPELINE
				for (int n = 0; n < 64; n++){
					int w_index = c + r * F + (n + m * 64) * FILTER_SIZE;
					w_buff[n][r][c][m] = weight[w_index];
				}
			}
		}
	}

	for (int x = 0; x < 14; x++){
		for (int y = 0; y < 14; y++){
			int mac_num = 0;
			for (int c = 0; c < F; c++){
				for (int r = 0; r < F; r++){
#pragma HLS PIPELINE rewind
					if (if_mac(x + c, y + r, 18)){
						mac_num++;
						for (int n = 0; n < 64; n++){
							bit32_t tmp = w_buff[n][r][c] ^ input[y + r - PADDING / 2][x + c - PADDING / 2];
							for (int i = 0; i < 32; i++){
								if (tmp[i] == 0)
									count[n]++;
							}
						}
					}
				}
			}
			for (int n = 0; n < 64; n++){
#pragma HLS UNROLL
				int tmp = 2 * count[n] - 32 * mac_num;
				output[n][y][x] = (tmp * con * k[n] + h[n]).is_neg() ? 0 : 1;
				count[n] = 0;
			}
		}
	}
}

void reshape(float* input, float* output) {
	for (int c = 0; c < 64; c++) {
		for (int y = 0; y < 7; y++) {
			for (int x = 0; x < 7; x++) {
				int o_index = c + (x + y * 7 ) * 64;
				int i_index = x + y * 7 + c * 49;
				output[o_index] = input[i_index];
			}
		}
	}
}

void dense(float* input, float* output, const float* weight, const float* bias, int M, int N, bool use_relu){
	float var_w = 2. / M;
	float c = sqrt(var_w);

	for (int n = 0; n < N; n++){
		float one_out = 0;
		for (int m = 0; m < M; m++) {
			int w_index = m * N + n;
			//output[n] += input[m] * weight[w_index] * c;
			one_out += (input[m] == weight[w_index]) ? 1 : 0; //XNOR
		}
		output[n] = (2 * one_out - M)*c;
		float biased = output[n] + bias[n];
		if (use_relu) output[n] = (biased > 0) ? 1 : 0;
		else output[n] = biased;
	}

}
