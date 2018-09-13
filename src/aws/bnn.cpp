#include "model_conv.h"
#include "model_dense.h"

inline bool if_mac(int x, int y, int I)
{
	return !(x < PADDING / 2 || x >= (I - PADDING / 2) || y < PADDING / 2 || y >= (I - PADDING / 2));
}

inline void load_weight(bit w_buff[32][5][5], const bit w[MAX_W_CONV])
{
	for (int wn = 0; wn < 32; wn++){
		for (int i = 0; i < F; i++){
			for (int j = 0; j < F; j++){
#pragma HLS PIPELINE rewind
				int w_index = i + j * F + wn * FILTER_SIZE;
				w_buff[wn][j][i] = w[w_index];
			}
		}
	}
}

inline void store_ofmap(fix o_buff[32][28][28], bit64_t out[28][28])
{
#pragma HLS ARRAY_PARTITION variable=k1 complete
#pragma HLS ARRAY_PARTITION variable=h1 complete
	for (int i = 0; i < 28; i++){
		for (int j = 0; j < 28; j++){
#pragma HLS PIPELINE
			for (int fn = 0; fn < 32; fn++){
#pragma HLS UNROLL
				fix tmp = o_buff[fn][j][i]*k1[fn]+h1[fn];
				out[j][i][fn] = tmp.is_neg() ? 0 : 1;
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
void conv_1(bit input[28][28], bit64_t output[28][28], const bit weight[MAX_W_CONV], fix con)
{
	fix output_buffer[32][28][28];
#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=1
	fix calc_buffer[32];
#pragma HLS ARRAY_PARTITION variable=calc_buffer complete
	bit weight_buffer[32][5][5];
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0
	bit line_buffer[F][28];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
	bit window_buffer[F][F];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

	for (int n = 0; n < 32; n++)
#pragma HLS UNROLL
		calc_buffer[n] = 0;

	for (int j = 0; j < I_WIDTH1; j++){
		for (int i = 0; i < 3; i++)
#pragma HLS PIPELINE
			line_buffer[i+2][j] = input[i][j];
	} //initialize linebuffer
	for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE
		for (int i = 2; i < F; i++) {
			window_buffer[i][j + 2] = line_buffer[i][j];
			if (i < F - 1)
				line_buffer[i][j] = line_buffer[i + 1][j];
			else
				line_buffer[i][j] = input[3][j];
		}
	} //initialize windowbuffer

	load_weight(weight_buffer, weight);
x:
	for (int xy = 0; xy < I_WIDTH1*I_WIDTH1; xy++){
#pragma HLS PIPELINE
#pragma AP dependence variable=line_buffer inter false
		int x = xy%I_WIDTH1;
		int y = xy / I_WIDTH1;
		int next_x = (x + 3) % 28;
	c:
		for (int c = 0; c < F; c++){
		r:
			for (int r = 0; r < F; r++){
				if (if_mac(x + c, y + r, 32)){
					for (int n = 0; n < 32; n++){
						fix tmp_out;
						tmp_out = (window_buffer[r][c] == weight_buffer[n][r][c]) ? con : con.getNeg();
						calc_buffer[n] += tmp_out;
					}
				}
			}
		}
		for (int nc = 0; nc < 32; nc++){
			output_buffer[nc][y][x] = calc_buffer[nc];
			calc_buffer[nc] = 0;
		}
		for (int i = 0; i < F; i++){
			for (int j  = 0; j < F - 1; j++)
				window_buffer[i][j] = window_buffer[i][j + 1]; //shift windowbuffer
			window_buffer[i][4] = line_buffer[i][next_x]; //shift next into window buffer
		}
		for (int i = 0; i < F-1; i++)
			line_buffer[i][next_x] = line_buffer[i+1][next_x];
		line_buffer[F-1][next_x] = (y + 3<I_WIDTH1) ? input[y + 3][next_x] : bit(0);
	}
	store_ofmap(output_buffer, output);
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

inline void ld_wt(int n, bit32_t w_buff[8][5][5], const bit32_t w[MAX_W_CONV/32]){
	for (int wn = 0; wn < 8; wn++){
		for (int r = 0; r < F; r++){
			for (int c = 0; c < F; c++){
#pragma HLS PIPELINE			
				int w_index = c + r * F + (n + wn) * FILTER_SIZE;
				w_buff[wn][r][c] = w[w_index];
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
	x = x - (((x) >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + (((x) >> 2) & 0x3333333333333333ULL);
    x = (x + ((x) >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	x = x + (x >> 8);
	x = x + (x >> 16);
	x = x + (x >> 32);
    return x & 0x000000000000007F;
}

void conv_2(bit64_t input[14][14], bit64_t output[28][28], const bit32_t weight[MAX_W_CONV/32]){
#pragma HLS ARRAY_PARTITION variable=k2 complete
#pragma HLS ARRAY_PARTITION variable=h2 complete

	bit32_t w_buff[8][5][5];
#pragma HLS ARRAY_PARTITION variable=w_buff complete dim=0
	bit32_t line_buff[F][I_WIDTH2];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1
	bit32_t window_buff[F][F];
#pragma HLS ARRAY_PARTITION variable=window_buff complete dim=0
	int count[8];
#pragma HLS ARRAY_PARTITION variable=count complete
	for (int n = 0; n < 8; n++)
#pragma HLS UNROLL
		count[n] = 0;

	int mac_num = 0;

	for (int n = 0; n < 64; n += 8){
		ld_wt(n, w_buff, weight);
		for (int j = 0; j < I_WIDTH2; j++)
			for (int i = 0; i < 3; i++)
#pragma HLS PIPELINE
				line_buff[i+2][j] = input[i][j](31, 0);
		for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE
			for (int i = 2; i < F; i++) {
				window_buff[i][j + 2] = line_buff[i][j];
				if (i < F - 1)
					line_buff[i][j] = line_buff[i + 1][j];
				else
					line_buff[i][j] = input[3][j](31, 0);
			}
		} //initialize windowbuffer
		for (int xy = 0; xy < I_WIDTH2*I_WIDTH2; xy++) {
#pragma HLS PIPELINE
#pragma AP dependence variable=line_buff inter false
			int x = xy%I_WIDTH2;
			int y = xy / I_WIDTH2;
			int next_x = (x + 3) % 14;
			for (int c = 0; c < F; c++) {
				for (int r = 0; r < F; r++) {
					if (if_mac(x + c, y + r, 18)) {
						mac_num++;
						for (int nn = 0; nn < 8; nn++) {
							bit32_t tmp = w_buff[nn][r][c] ^ window_buff[r][c];
							count[nn] += popcount(tmp.to_uint());
						}
					}
				}
			}
			for (int i = 0; i < F; i++) {
				for (int j = 0; j < F - 1; j++)
					window_buff[i][j] = window_buff[i][j + 1]; //shift window buffer
				window_buff[i][4] = line_buff[i][next_x]; //shift next into windowbuffer
			}
			for (int i = 0; i < F-1; i++) 
				line_buff[i][next_x] = line_buff[i + 1][next_x];
			line_buff[F-1][next_x] = (y + 3 < 14) ? input[y + 3][next_x](31, 0) : bit32_t(0);
			for (int nn = 0; nn < 8; nn++) {
				fix tmp = ((mac_num << 5) - (count[nn] << 1)) * k2[n + nn];
				output[y][x][n + nn] = (tmp + h2[n + nn]).is_neg() ? 0 : 1;
				count[nn] = 0;
			}
			mac_num = 0;
		}
	}
}

void ld_wt_fc1(bit64_t input[14][14], bit64_t i_buff[7], const bit64_t weight[MAX_W_FC/64], bit64_t w_buff[7]){

	static int m = 0;
	static int n = 0;
	int M = 7 * m;
	int MN = n * 49 + M;
	for (int i = 0; i < 7; i++){
#pragma HLS PIPELINE
		int w_index = MN + i;
		int i_index = M + i;
		w_buff[i] = weight[w_index];
		i_buff[i] = input[m][i];
	}
	m++;
	if (m == 7) {
		n++;
		if(n == FC2_UNITS)
			n = 0;
		m = 0;
	}
}

void calc_1(bit64_t w_buff[7], bit64_t i_buff[7], bit64_t output[8], const fix con){

	static int m = 0;
	static int n = 0;
	static int count = 0;
	static bit64_t tmp_out = 0;
	int M = 7 * m;
	m++;
	for (int mm = 0; mm < 7; mm++){
#pragma HLS UNROLL
		bit64_t tmp = w_buff[mm] ^ i_buff[mm];
		count += (64 - popcount(tmp.to_uint64()));
	}
	if (m == 7){
		int calc_result = (count << 1) - FC1_UNITS;
		tmp_out[n & 63] = (calc_result * con + b_fc1[n]).is_neg() ? 0 : 1;
		count = 0;
		n++;
		if ((n & 63) == 0) {
			output[(n >> 6) - 1] = tmp_out;
			tmp_out = 0;
			if (n & 512)
				n = 0;
		}
		m = 0;
	}
}

void dense_1(bit64_t input[14][14], bit64_t output[8], const bit64_t weight[MAX_W_FC/64], const fix con){
#pragma HLS INLINE off
	bit64_t w_buff[7];
#pragma HLS ARRAY_PARTITION variable=w_buff complete
	bit64_t i_buff[7];
#pragma HLS ARRAY_PARTITION variable=i_buff complete
	/*for (int i = 0; i < 7; i++)
#pragma HLS UNROLL
		w_buff[i]=0;*/

	for (int i = 0; i < FC2_UNITS*7; i++){
#pragma HLS DATAFLOW
		ld_wt_fc1(input, i_buff, weight, w_buff);
		calc_1(w_buff, i_buff, output, con);
	}
}

void ld_wt_fc2(int n, const bit64_t weight[80], bit64_t w_buff[8]){
	for (int i = 0; i < 8; i++){
#pragma HLS PIPELINE
		int w_index = (n << 3) + i;
		w_buff[i] = weight[w_index];
	}
}

void dense_2(bit64_t input[8], fixo output[10], const bit64_t weight[80], const fix con){
#pragma HLS INLINE off
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
		output[n] = calc_result * con + b_fc2[n];
		count = 0;
	}
}

#ifdef __cplusplus
extern "C" {
#endif
void bnn(bit8_t x[I_WIDTH1 * I_WIDTH1], fixo output[10]){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	bit mem1[28][28];
	bit64_t mem2[28][28];
	bit64_t mem3[14][14];
	bit64_t memfc2[8];
#pragma HLS ARRAY_PARTITION variable=memfc2 complete

	for (int i = 0; i < I_WIDTH1; i++)
		for (int j = 0; j < I_WIDTH1; j++){
#pragma HLS PIPELINE
			int i_index = j + i*I_WIDTH1;
			mem1[i][j] = x[i_index];
		}

	conv_1(mem1, mem2, w_conv1, con1);

	max_pool(mem2, mem3, 32, I_WIDTH1);

	conv_2(mem3, mem2, w_conv2);

	max_pool(mem2, mem3, 64, I_WIDTH2);

	dense_1(mem3, memfc2, w_fc1, con_fc1);

	dense_2(memfc2, output, w_fc2, con_fc2);

}
#ifdef __cplusplus
}
#endif
