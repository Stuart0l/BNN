#===============================================================#
#                                                               #
#                       	run1.sh                         	#
#                                                               #
#   	A bash script to run the software emulation 			#
#																#
#                                                               #
#===============================================================#


#!/bin/bash

# set up emulation configuration
echo "#################################################"
echo " Setting emulation configuration..."
echo "#################################################"
export  XCL_EMULATION_MODE=true
emconfigutil --platform=$AWS_PLATFORM


### COMPILATION
# create some blank-line space for easy readability
echo ""; echo ""; echo "" ; echo ""
echo "####################################################"
echo " Compiling project with K_CONST=$k_value"
echo "####################################################"
make clean
make ocl OCL_TARGET=sw_emu OCL_PLATFORM=$AWS_PLATFORM K_CONST=3

### EXECUTION
echo ""; echo ""; echo "" ; echo ""
echo "####################################################"
echo " Executing DigitRec with K_CONST=$k_value"
echo "####################################################"
./BNN_host.exe -f BNN.sw_emu.xclbin 



