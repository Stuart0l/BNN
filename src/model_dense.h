#ifndef MODEL_DENSE
#define MODEL_DENSE

#include"layer.h"

const fix b_fc1[FC2_UNITS] = {
#include"../data/weight_11p"
};

const fix b_fc2[OUT] = {
#include"../data/weight_13p"
};

#endif
