# Set kernel name
KERNEL_NAME = BNN

# Set host source and headers
HOST_SRC_CPP = ./src/host/layer.cpp ./src/host/test_bnn.cpp ./src/host/utils.cpp 
HOST_SRC_H   = ./src/host/layer.h ./src/host/model_conv.h ./src/host/model_dense.h ./src/host/typedef.h ./src/host/utils.h
DATA         = ./data/*

# Set host code include paths
HOST_INC = -I$(XILINX_SDX)/Vivado_HLS/include 
HOST_LIB = -L$(XILINX_SDX)/Vivado_HLS/lib

# Set kernel file
OCL_KERNEL_SRC = ./src/kernel/bnn.cpp 
OCL_KERNEL_H = ./src/host/typedef.h 

SDSOC_KERNEL_SRC = ./src/kernel/bnn.cpp 
SDSOC_KERNEL_H = ./src/host/typedef.h 

SW_KERNEL_SRC = ./src/kernel/bnn.cpp 
SW_KERNEL_H = ./src/host/typedef.h  ./src/host/bnn.h

# Set opencl kernel arguments   
OCL_KERNEL_ARGS = --max_memory_ports all 
#--report system 



#-------------------------
# Leave the rest to harness
#-------------------------
include ../harness/harness.mk
