#include "../host/typedef.h"
#include "../host/model_conv.h"
#include <cstdio>
#include <stdint.h>
#include <iostream>

using namespace std;

void pad(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I) {
  const int ifmap_size = I * I;
  const int ofmap_size = (I+PADDING) * (I+PADDING);

  for (int i = 0; i < MAX_FMAP; i++) output[i] = 0;

  for (int m = 0; m < 64 && m<M; m++) {
    for (int x = 0; x < 32 && x<I; x++) {
      for (int y = 0; y < 32 && y<I; y++) {
        int i_index = x + y * 32 + m * 1024;
        int o_index = (x + PADDING/2) + (y + PADDING/2)* 32 + m * 1024;
        output[o_index] = input[i_index];
      }
    }
  }
}

inline bool if_mac(int x, int y, int I)
{
  if (x < PADDING / 2 || x >= (I - PADDING / 2) || y < PADDING / 2 || y >= (I - PADDING / 2))
    return false;
  return true;
}


inline void load_weight(int n, int m, int N, bit w_buff[8][8][5][5], const bit w[MAX_W_CONV])
{
    for (int wn = 0; wn < 8; wn++)
    {
        for (int wm = 0; wm < 8; wm++)
        {
            for (int i = 0; i < F; i++)
            {
                for (int j = 0; j < F; j++)
                {
                    int w_index = i + j * F + (n + wn + (m + wm) * N) * FILTER_SIZE;
                    w_buff[wm][wn][j][i] = w[w_index];
                }
            }
        }
    }
}

inline void load_ifmap(int m, bit i_buff[8][32][32], bit in[MAX_FMAP])
{
    for (int fm = 0; fm < 8; fm++)
    {
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                int i_index = i + j * 32 + (m + fm) * 1024;
                i_buff[fm][j][i] = in[i_index];
            }
        }
    }
}

inline void store_ofmap(int n, fix o_buff[8][32][32], fix out[MAX_FMAP])
{
    for (int fn = 0; fn < 8; fn++)
    {
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                int o_index = i + j * 32 + (n + fn) * 1024;
                out[o_index] = o_buff[fn][j][i];
                o_buff[fn][j][i] = 0.; // necessary? also doing this in conv_2d
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
void conv_2d(bit input[MAX_FMAP], fix output[MAX_FMAP], const bit weight[MAX_W_CONV], int M, int N, int I, fix con)
{
  
    bit input_buffer[8][32][32];
#pragma HLS ARRAY_PARTITION variable=input_buffer block factor=8 dim=1
    fix output_buffer[8][32][32];
#pragma HLS ARRAY_PARTITION variable=output_buffer block factor=8 dim=1
    bit weight_buffer[8][8][5][5];
#pragma HLS ARRAY_PARTITION variable=weight_buffer block factor=8 dim=1
    
  int O = I - F + 1;
  int ifmap_size = I * I;
  int ofmap_size = O * O;
   /*
    fix const2 = 2.0;
    fix var_w = const2 / (F*F * M);  //convert F, M to fixed point!
    fix con = hls::sqrt(var_w);
    */
  for (int i = 0; i < MAX_FMAP; i++) output[i] = 0;
    for (int n = 0; n < 8; n++)
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                output_buffer[n][i][j] = 0; //reset out_buffer to 0.
    
    
    for (int n = 0; n < N; n += 8)
    {
        for (int m = 0; m < M; m += 8)
        {
            load_weight(n, m, N, weight_buffer, weight);
            load_ifmap(m, input_buffer, input);
            for (int c = 0; c < F; c++)
            {
                for (int r = 0; r < F; r++)
                {
                    for (int x = 0; x < O; x++)
                    {
                        for (int y = 0; y < O; y++)
                        {
                            if (if_mac(x + c, y + r, I))
                            {
                            loop_nn:
                                for (int nn = 0; nn < 8; nn++)
                                {
#pragma HLS UNROLL
                                    if (nn + n < N)
                                    {
                                    loop_mm:
                                        for (int mm = 0; mm < 8; mm++)
                                        {
#pragma HLS UNROLL
                                            if (mm + m < M)
                                            {
                                                output_buffer[nn][y][x] += ((input_buffer[mm][y + r][x + c] == weight_buffer[mm][nn][r][c]) ? con : con.getNeg()); //do not use -, use getNeg
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        store_ofmap(n, output_buffer, output);
    }
}

void max_pool(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I){
  int O = I / 2;
  int ifmap_size = I * I;
  int ofmap_size = O * O;

  for (int i = 0; i < MAX_FMAP; i++) output[i] = 0;

  for (int m = 0; m < 64 && m<M ; m++){
    for (int x = 0; x < 32 && x<O ; x++){
      for (int y = 0; y < 32 && y<O ; y++){
        int o_index = x + y * 32 + m * 1024;
        bit max = 0;
        for (int c = 0; c < 2; c++){
          for (int r = 0; r < 2; r++){
            int i_index = 2 * x + c + (2 * y + r) * 32 + m * 1024;
            if (input[i_index] < 0) max = input[i_index]; //
          }
        }
        output[o_index] = max;
      }
    }
  }
}

void batch_norm(fix input[MAX_FMAP], bit output[MAX_FMAP], const fix miu[MAX_F], const float sigma[MAX_F], const fix gamma[MAX_F], const fix beta[MAX_F], int M, int I){
  int ifmap_size = I * I;

  for (int m = 0; m < 64; m++){
      fix s = sqrt(sigma[m] + 0.00001); //??
      fix k = gamma[m] / s;
      fix tmp_miu = miu[m];
      fix h = tmp_miu.getNeg() * gamma[m] / s + beta[m];
      for (int x = 0; x < 32 || x < I; x++){
          for (int y = 0; y < 32 || y < I; y++){
            int index = x + y * 32 + m * 1024;
            output[index] = ((input[index] * k + h).is_neg() ? 0 : 1);
              
          }
        }
      }
    }

extern "C"
{
    void BNN(bit8_t x[I_WIDTH1 * I_WIDTH1], bit8_t output[O_WIDTH*O_WIDTH * 64]){
        

        #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
        #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
        #pragma HLS INTERFACE s_axilite port=x bundle=control
        #pragma HLS INTERFACE s_axilite port=output bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control
        
        
        bit mem_conv1[MAX_FMAP] = {0};
        bit mem_conv2[MAX_FMAP] = {0};
        fix mem_conv3[MAX_FMAP] = {0};
        
        for (int i = 0; i < I_WIDTH1; i++)
            for (int j = 0; j < I_WIDTH1; j++){
                int i_index = i + j*I_WIDTH1;
                int o_index = i + j*32;
                mem_conv2[o_index] = x[i_index];
            }

        pad(mem_conv2, mem_conv1, 1, I_WIDTH1);
        
        conv_2d(mem_conv1, mem_conv3, w_conv1, 1, 32, 32, con1);
        
        batch_norm(mem_conv3, mem_conv1, miu1, sigma1, gamma1, beta1, 32, I_WIDTH1);

        max_pool(mem_conv1, mem_conv2, 32, I_WIDTH1);

        for (int i = 0; i < MAX_FMAP; i++) mem_conv1[i] = 0;

        pad(mem_conv2, mem_conv1, 32, I_WIDTH2);

        conv_2d(mem_conv1, mem_conv3, w_conv2, 32, 64, 18, con2);

        batch_norm(mem_conv3, mem_conv1, miu2, sigma2, gamma2, beta2, 64, I_WIDTH2);

        max_pool(mem_conv1, mem_conv2, 64, I_WIDTH2);
        
        //can be unrolled
        for (int m = 0; m < 64; m++)
            for (int i = 0; i < O_WIDTH; i++)
                for (int j = 0; j < O_WIDTH; j++){
                    int i_index = i + j * 32 + m * 1024;
                    int o_index = i + j * O_WIDTH + m * O_WIDTH*O_WIDTH;
                    output[o_index] = mem_conv2[i_index];
                }
        
    }
}
