KERNEL_NAME = bnn

HOST_SRC_CPP = host.cpp
HOST_SRC_H = bnn.h typedef.h timer.h

HOST_INC = -I$(XILINX_SDX)/Vivado_HLS/include
HOST_LIB = -L$(XILINX_SDX)/Vivado_HLS/lib

OCL_KERNEL_SRC = bnn.cpp
OCL_KERNEL_SRC_H = model_conv.h model_dense.h typedef.h

OCL_KERNEL_ARGS = --max_memory_ports all

include ../harness/harness.mk