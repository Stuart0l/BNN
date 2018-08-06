#ifndef MODEL_CONV
#define MODEL_CONV

#include"layer.h"

const bit w_conv1[MAX_W_CONV] = {
    #include"../data/weight_0b"
}; //binary weight

/* const fix miu1[MAX_F] = {
    #include"data/weight_3p"
};

const float sigma1[MAX_F] = {
    #include"data/weight_4p"
};

const fix gamma1[MAX_F] = {
    #include"data/weight_1p"
};

const fix beta1[MAX_F] = {
    #include"data/weight_2p"
}; */

const fix k1[MAX_F] = {
    #include"../data/weight_k1"
};

const fix h1[MAX_F] = {
    #include"../data/weight_h1"
};

const bit w_conv2[MAX_W_CONV] = {
    #include"data/weight_5b"
}; //binary weight

/* const fix miu2[MAX_F] = {
    #include"data/weight_8p"
};

const float sigma2[MAX_F] = {
    #include"data/weight_9p"
};

const fix gamma2[MAX_F] = {
    #include"data/weight_6p"
};

const fix beta2[MAX_F] = {
    #include"data/weight_7p"
}; */

const fix k2[MAX_F] = {
    #include"../data/weight_k2"
};

const fix h2[MAX_F] = {
    #include"../data/weight_h2"
};

#endif
