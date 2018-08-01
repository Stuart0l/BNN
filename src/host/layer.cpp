#include "layer.h"
#include "typedef.h"
#include <cmath>
#include <iostream>

using namespace std;

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

