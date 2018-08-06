#ifndef BNN
#define BNN

#include"layer.h"
#include"typedef.h"

void pad(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I);

void conv_2d(bit input[MAX_FMAP], fix output[MAX_FMAP], const bit weight[MAX_W_CONV], int M, int N, int I, fix con);

void max_pool(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I);

void batch_norm(fix input[MAX_FMAP], bit output[MAX_FMAP], const fix miu[MAX_F], const fix sigma[MAX_F], const fix gamma[MAX_F], const fix beta[MAX_F], int M, int I);

extern "C"{
void BNN(bit x[O_WIDTH*O_WIDTH * 64]);
}
#endif
