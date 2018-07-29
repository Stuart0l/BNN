#include "../host/layer.h"
#include "../host/model_conv.h"
#include "../host/typedef.h"
#include <fstream>
#include <iostream>
#include <iomanip>
//#include <hls_math.h>
//using namespace std;
extern "C"
{
void BNN (bit* x, bit* output){
    //(bit x[I_WIDTH1 * I_WIDTH1], bit output[O_WIDTH*O_WIDTH * 64]){
    

#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
    
  bit mem_conv1[MAX_FMAP] = {0};
  bit mem_conv2[MAX_FMAP] = {0};
  fix mem_conv3[MAX_FMAP] = { 0 }; //0.0?

    for (int i = 0; i < I_WIDTH1; i++)
        for (int j = 0; j < I_WIDTH1; j++){
            int i_index = i + j*I_WIDTH1;
            int o_index = i + j*32;
            mem_conv2[o_index] = x[i_index];
        } //why use buffer?

  pad(mem_conv2, mem_conv1, 1, I_WIDTH1);
    
  fix con1 = 0.28284271;
  conv_2d(mem_conv1, mem_conv3, w_conv1, 1, 32, 32, con1);
 
  batch_norm(mem_conv3, mem_conv1, miu1, sigma1, gamma1, beta1, 32, I_WIDTH1);

  max_pool(mem_conv1, mem_conv2, 32, I_WIDTH1);

  for (int i = 0; i < MAX_FMAP; i++) mem_conv1[i] = 0;

  pad(mem_conv2, mem_conv1, 32, I_WIDTH2);

  fix con2 = 0.05;
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
