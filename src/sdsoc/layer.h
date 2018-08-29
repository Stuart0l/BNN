#ifndef LAYER_H
#define LAYER_H

#include"typedef.h"

const int I_WIDTH1 = 28; //conv1 input width
const int I_WIDTH2 = 14; //conv2 input width
const int O_WIDTH = 7; //conv output width
const int F = 5; //filter width
const int FILTER_SIZE = F*F;
const int PADDING = F - 1;
const int MAX_FMAP = 65536; //32*32*64
const int MAX_WIDTH = 28;
const int MAX_F = 64; //{32, 64} num of conv2 output fmaps
const int MAX_W_CONV = 51200;//5*5*32*64 num of conv2 weights
const int FC1_UNITS = O_WIDTH*O_WIDTH * 64; //num of fc1 input units
const int FC2_UNITS = 512;
const int MAX_W_FC = FC1_UNITS*FC2_UNITS;
const int OUT = 10;
const fix con1 = sqrt(2.0/(F*F*1));//0.28284271;
const fix con2 = sqrt(2.0/(F*F*32));//0.05;
const fix con_fc1 = sqrt(2.0/FC1_UNITS);
const fix con_fc2 = sqrt(2.0/FC2_UNITS);

#endif
