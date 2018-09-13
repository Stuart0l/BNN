#include "model_conv.h"
#include "model_dense.h"
#include "bnn.h"
#include <fstream>
#include <iomanip>

inline bool if_mac(int x, int y, int I)
{
	if (x < PADDING / 2 || x >= (I - PADDING / 2) || y < PADDING / 2 || y >= (I - PADDING / 2))
		return false;
	return true;
}

inline void load_weight(int N, bit w_buff[32][5][5], const bit w[MAX_W_CONV])
{
	for (int wn = 0; wn < 32; wn++){
		for (int i = 0; i < F; i++){
			for (int j = 0; j < F; j++){
#pragma HLS PIPELINE
				int w_index = i + j * F + wn * FILTER_SIZE;
				w_buff[wn][j][i] = w[w_index];
			}
		}
	}
}

inline void store_ofmap(fix o_buff[32][28][28], bit64_t out[28][28], const fix k[MAX_F], const fix h[MAX_F])
{
	for (int i = 0; i < 28; i++){
		for (int j = 0; j < 28; j++){
#pragma HLS PIPELINE
			for (int fn = 0; fn < 32; fn++){
#pragma HLS UNROLL
				fix tmp = o_buff[fn][j][i]*k[fn]+h[fn];
				out[j][i][fn] = tmp.is_neg() ? 0 : 1;
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
void conv_1(bit input[28][28], bit64_t output[28][28], const bit weight[MAX_W_CONV], const fix k[MAX_F], const fix h[MAX_F], int M, int N, int I, fix con)
{
	fix output_buffer[32][28][28];
#pragma HLS ARRAY_PARTITION variable = output_buffer complete dim = 1
	fix calc_buffer[32];
#pragma HLS ARRAY_PARTITION variable = calc_buffer complete
	bit weight_buffer[32][5][5];
#pragma HLS ARRAY_PARTITION variable = weight_buffer complete dim = 1
	int O = I - F + 1;

	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			for (int n = 0; n < 32; n++)
#pragma HLS UNROLL
				output_buffer[n][i][j] = 0;

	for (int n = 0; n < 32; n++)
#pragma HLS UNROLL
		calc_buffer[n] = 0;

	load_weight(N, weight_buffer, weight);
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
						for (int n = 0; n < 32; n++){
							fix tmp_out;
							if (n < N){
								tmp_out = ((input[y + r - PADDING / 2][x + c - PADDING / 2] == weight_buffer[n][r][c]) ? con : con.getNeg()); //do not use -, use getNeg
							}
							calc_buffer[n] += tmp_out;
						}
					}
				}
			}
			for (int nc = 0; nc < 32; nc++){
#pragma HLS UNROLL
				output_buffer[nc][y][x] += calc_buffer[nc];
				calc_buffer[nc] = 0;
			}
		}
	}
	store_ofmap(output_buffer, output, k, h);
}

void max_pool(bit64_t input[28][28], bit64_t output[14][14], int M, int I){
	int O = I / 2;

	for (int j = 0; j < 14; j++)
		for (int i = 0; i < 14; i++)
			output[j][i] = 0;

	for (int x = 0; x < O; x++){
		for (int y = 0; y < O; y++){
			bit64_t max =
			input[2*y+0][2*x+0]|
			input[2*y+1][2*x+0]|
			input[2*y+0][2*x+1]|
			input[2*y+1][2*x+1];
			output[y][x] = max;
		}
	}
}

inline void ld_wt(int n, bit32_t w_buff[16][5][5], const bit w[MAX_W_CONV]){
	for (int m = 0; m < 32; m++){
		for (int c = 0; c < F; c++){
			for (int r = 0; r < F; r++){
#pragma HLS PIPELINE
				for (int wn = 0; wn < 16; wn++){
					int w_index = c + r * F + (n + wn + m * 64) * FILTER_SIZE;
					w_buff[wn][r][c][m] = w[w_index];
				}
			}
		}
	}
}

inline int popcount(unsigned int x){
#pragma HLS INLINE
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0f0f0f0f;
	x = x + (x >> 8);
	x = x + (x >> 16);
	return x & 0x0000003f;
} //Hecker Popcount

inline int popcount(unsigned long long x){
#pragma HLS INLINE
	x = x - (((x) >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + (((x) >> 2) & 0x3333333333333333ULL);
    x = (x + ((x) >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	x = x + (x >> 8);
	x = x + (x >> 16);
	x = x + (x >> 32);
    return x & 0x000000000000007F;
}

void conv_2(bit64_t input[14][14], bit64_t output[28][28], const bit weight[MAX_W_CONV], const fix k[MAX_F], const fix h[MAX_F]){

	bit32_t w_buff[16][5][5];
#pragma HLS ARRAY_PARTITION variable=w_buff complete dim=1
	bit o_buff[16][14][14];
#pragma HLS ARRAY_PARTITION variable=o_buff complete dim=1
	int count[16];
#pragma HLS ARRAY_PARTITION variable=count complete
	for (int n = 0; n < 16; n++)
#pragma HLS UNROLL
		count[n] = 0;

	int mac_num = 0;

	for (int n = 0; n < 64; n += 16){
		ld_wt(n, w_buff, weight);
		for (int x = 0; x < 14; x++){
			for (int y = 0; y < 14; y++){
				for (int c = 0; c < F; c++){
					for (int r = 0; r < F; r++){
#pragma HLS PIPELINE rewind
						if (if_mac(x + c, y + r, 18)){
							mac_num++;
							for (int nn = 0; nn < 16; nn++){
								bit32_t tmp = w_buff[nn][r][c] ^ input[y + r - PADDING / 2][x + c - PADDING / 2];
								count[nn] += (32 - popcount(tmp.to_uint()));
							}
						}
					}
				}
				for (int nn = 0; nn < 16; nn++){
#pragma HLS UNROLL
					int tmp = (count[nn] << 1) - (mac_num << 5);
					output[y][x][n + nn] = (tmp * k[n + nn] + h[n + nn]).is_neg() ? 0 : 1;
					count[nn] = 0;
				}
				mac_num = 0;
			}
		}
	}
}

void ld_wt_fc1(int m, int n, bit64_t weight[MAX_W_FC/64], bit64_t w_buff[8]){
#pragma HLS INLINE
	int MN = n * 49 + m * 7;
	if (m < 7){
		for (int i = 0; i < 7; i++){
#pragma HLS PIPELINE
			int w_index = MN + i;
			w_buff[i] = weight[w_index];
		}
	}
}

void calc_1(int m, int n, bit64_t w_buff[7], bit64_t input[49], bit64_t output[8], int *count, const fix bias[FC2_UNITS], const fix con){
	for (int mm = 0; mm < 7; mm++){
#pragma HLS UNROLL
		bit64_t tmp = w_buff[mm] ^ input[7*m+mm];
		*count += (64 - popcount(tmp.to_uint64()));
	}
	if (m == 6){
		int calc_result = (*count << 1) - FC1_UNITS;
		output[n >> 6][n & 63] = (calc_result * con + bias[n]).is_neg() ? 0 : 1;
		*count = 0;
	}
}

void dense_1(bit64_t input[49], bit64_t output[8], bit64_t weight[MAX_W_FC/64], const fix bias[FC2_UNITS], const fix con){
	bit64_t w_buff[7];
#pragma HLS ARRAY_PARTITION variable=w_buff complete
	int count;

	for (int n = 0; n < FC2_UNITS; n++){
		for (int m = 0; m < 7; m++){
			ld_wt_fc1(m, n, weight, w_buff);
			calc_1(m, n, w_buff, input, output, &count, bias, con);
		}
	}
}

void ld_wt_fc2(int n, bit64_t weight[80], bit64_t w_buff[8]){
	for (int i = 0; i < 8; i++){
#pragma HLS PIPELINE
		int w_index = (n << 3) + i;
		w_buff[i] = weight[w_index];
	}
}

void dense_2(bit64_t input[8], fixo output[10], bit64_t weight[80], const fix bias[OUT], const fix con){
	bit64_t w_buff[8];
#pragma HLS ARRAY_PARTITION variable=w_buff complete
	int count = 0;

	for (int n = 0; n < 10; n++){
		ld_wt_fc2(n, weight, w_buff);
		for (int m = 0; m < 8; m++){
#pragma HLS UNROLL
			bit64_t tmp = w_buff[m] ^ input[m];
			count += (64 - popcount(tmp.to_uint64()));
		}
		int calc_result = (count << 1) - FC2_UNITS;
		output[n] = calc_result * con + bias[n];
		count = 0;
	}
}

void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], fixo output[10], bit64_t weight1[MAX_W_FC/64], bit64_t weight2[80]){

#pragma HLS ARRAY_PARTITION variable=k1 complete
#pragma HLS ARRAY_PARTITION variable=h1 complete
#pragma HLS ARRAY_PARTITION variable=k2 complete
#pragma HLS ARRAY_PARTITION variable=h2 complete
	bit mem1[28][28];
	bit64_t mem2[28][28];
	bit64_t mem3[14][14];
#pragma HLS ARRAY_PARTITION variable=mem3 complete dim=1
	bit64_t memfc1[49];
#pragma HLS ARRAY_PARTITION variable=memfc1 cyclic factor=7
	bit64_t memfc2[8];
#pragma HLS ARRAY_PARTITION variable=memfc2 complete

	for (int i = 0; i < I_WIDTH1; i++)
		for (int j = 0; j < I_WIDTH1; j++){
#pragma HLS PIPELINE
			int i_index = j + i*I_WIDTH1;
			mem1[i][j] = x[i_index];
		}

	conv_1(mem1, mem2, w_conv1, k1, h1, 1, 32, 32, con1);

	max_pool(mem2, mem3, 32, I_WIDTH1);

	conv_2(mem3, mem2, w_conv2, k2, h2);

	max_pool(mem2, mem3, 64, I_WIDTH2);

	for (int i = 0; i < O_WIDTH; i++)
		for (int j = 0; j < O_WIDTH; j++){
#pragma HLS PIPELINE
			int o_index = i + j * O_WIDTH;
			memfc1[o_index] = mem3[j][i];
		}
#pragma HLS ARRAY_PARTITION variable=b_fc1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=b_fc2 complete
	dense_1(memfc1, memfc2, weight1, b_fc1, con_fc1);

	dense_2(memfc2, output, weight2, b_fc2, con_fc2);
}
