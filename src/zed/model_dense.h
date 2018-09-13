#ifndef MODEL_DENSE
#define MODEL_DENSE

#include"layer.h"

const fix b_fc1[FC2_UNITS] = {
#include"../../data/weight_bfc1"
};

const fix b_fc2[OUT] = {
#include"../../data/weight_bfc2"
};

#endif
