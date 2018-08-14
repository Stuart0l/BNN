#ifndef TYPEDEF
#define TYPEDEF

#include<ap_int.h>
#include<ap_fixed.h>

#define NUMBER_OF_BITS 24
#define NUMBER_OF_FRACTION_DIGITS 16

typedef ap_int<1> bit;
typedef char bit8_t;
typedef ap_int<32> bit32_t;
typedef ap_int<64> bit64_t;
typedef ap_fixed<NUMBER_OF_BITS, (NUMBER_OF_BITS - NUMBER_OF_FRACTION_DIGITS), AP_RND> fix;

#endif
