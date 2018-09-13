#ifndef MODEL_CONV
#define MODEL_CONV

#include"layer.h"

const bit w_conv1[MAX_W_CONV] = {
    #include"../../data/weight_conv1"
}; //binary weight

const fix k1[MAX_F] = {
    #include"../../data/weight_k1"
};

const fix h1[MAX_F] = {
    #include"../../data/weight_h1"
};

const bit w_conv2[MAX_W_CONV] = {
    #include"../../data/weight_conv2"
}; //binary weight

const fix k2[MAX_F] = {
    #include"../../data/weight_k2"
};

const fix h2[MAX_F] = {
    #include"../../data/weight_h2"
};

#endif
