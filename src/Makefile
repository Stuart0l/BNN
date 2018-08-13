# (c) Copyright 2013 - 2016 Xilinx, Inc. All rights reserved. 
# 
# This file contains confidential and proprietary information of Xilinx, Inc. and
# is protected under U.S. and international copyright and other intellectual
# property laws.
# 
# DISCLAIMER 
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
# 
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in any
# application requiring fail-safe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
# 
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
# ALL TIMES. 

APPSOURCES = test_bnn.cpp bnn.cpp
EXECUTABLE = bnn.elf

HLSONLY = -mno-bitstream -mno-boot-files

PLATFORM = zed
SDSFLAGS = -sds-pf ${PLATFORM} \
	-sds-hw bnn bnn.cpp -hls-tcl hls.tcl \
  -clkid 2 -sds-end \
	-poll-mode 1

CC = sds++
GCC = g++

CFLAGS = -Wall -O3 -c
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
LFLAGS = -O3

MACADDRESS = 00:0a:35:00:1e:51

OBJECTS := $(APPSOURCES:.cpp=.o)
DEPS := $(OBJECTS:.o=.d)

.PHONY: all

all: ${EXECUTABLE}
	#../setconfig.sh; \
	cd sd_card; \
	../../change_mac.sh ${MACADDRESS}; \
  cp -r ../data .

sw:
	g++ -g -I/opt/Xilinx/Vivado/2018.1/include ${APPSOURCES} -o bnn_sw
	./bnn_sw

hls: ${OBJECTS}
	${CC} ${HLSONLY} ${SDSFLAGS} ${LFLAGS} ${OBJECTS} -o $@ 

${EXECUTABLE}: ${OBJECTS}
	${CC} ${SDSFLAGS} ${LFLAGS} ${OBJECTS} -o $@ 

-include ${DEPS}

%.o: %.cpp
	${CC} ${SDSFLAGS} ${CFLAGS} $< -o $@

clean:
	${RM} ${EXECUTABLE} ${OBJECTS} ${DEPS}

ultraclean: clean
	${RM} ${EXECUTABLE}.bit 
	${RM} -rf _sds sd_card
	${RM} hls hls.bit; ${RM} bnn_sw
