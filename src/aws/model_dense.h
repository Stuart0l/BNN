#ifndef MODEL_DENSE
#define MODEL_DENSE

#include"layer.h"

const bit64_t w_fc1[MAX_W_FC/64] = {
    #include"../../data/weight_fc1_bp"
};

const fix b_fc1[FC2_UNITS] = {
#include"../../data/weight_bfc1"
};

const bit64_t w_fc2[80] = {
    #include"../../data/weight_fc2_bp"
};

const fix b_fc2[OUT] = {
#include"../../data/weight_bfc2"
};

#endif
