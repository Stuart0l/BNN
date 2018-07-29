#ifndef TYPEDEF
#define TYPEDEF

#include<ap_int.h>
#include<ap_fixed.h>

#define NUMBER_OF_BITS 24
#define NUMBER_OF_FRACTION_DIGITS 16

typedef ap_int<1> bit;
typedef ap_fixed<NUMBER_OF_BITS,(NUMBER_OF_BITS-NUMBER_OF_FRACTION_DIGITS),AP_RND> fix;  //rounds to positive inf

#ifdef OCL
#include <string>
// target device
// change here to map to a different device
const std::string TARGET_DEVICE = "xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0";
#endif

#endif
