#ifndef MODEL_DENSE
#define MODEL_DENSE

#include"layer.h"

const float w_fc1[MAX_W_FC] = {
#include"data/weight_10b"
};

const float b_fc1[FC2_UNITS] = {
#include"data/weight_11p"
};

const float w_fc2[FC2_UNITS*OUT] = {
#include"data/weight_12b"
};

const float b_fc2[OUT] = {
#include"data/weight_13p"
};

#endif
