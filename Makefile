PLATFORM := zc702

# Run Target:
#   hw  - Compile for hardware
#   emu - Compile for emulation (Default)
TARGET := emu

TARGET_OS := linux

pwd := $(CURDIR)

COMMON_REPO := $(pwd)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

include $(ABS_COMMON_REPO)/sds_utils/sds_utils.mk

ADDL_FLAGS := -Wno-unused-label

# Emulation Mode:
#     debug     - Include debug data
#     optimized - Exclude debug data (Default)
EMU_MODE := optimized

EXCUTABLE := bnn.elf

BUILD_DIR := build/$(PLATFORM)_$(TARGET_OS)_$(TARGET)

SRC_DIR := src
OBJECTS += \
$(pwd)/$(BUILD_DIR)/bnn.o \
$(pwd)/$(BUILD_DIR)/test_bnn.o

HW_FLAGS := -sds-hw bnn bnn.cpp -sds-end

EMU_FLAGS := 
ifeq ($(TARGET), emu)
	EMU_FLAGS := -mno-bitstream -mno-boot-files -emulation $(EMU_MODE)
endif

IFLAGS := -I.
CFLAGS = -Wall -O3 -c
CFLAGS += -MT"$@" -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" 
CFLAGS += -I$(sds_utils_HDRS)
CFLAGS += $(ADDL_FLAGS)
LFLAGS = "$@" "$<" 

SDSFLAGS := -sds-pf $(PLATFORM) \
	-target-os $(TARGET_OS)

CC := sds++ $(SDSFLAGS)

.PHONY: all
all: ${BUILD_DIR}/${EXCUTABLE}

${BUILD_DIR}/${EXCUTABLE}: ${OBJECTS}
	mkdir -p ${BUILD_DIR}
	@echo 'Building Target: $@'
	@echo 'Trigerring: SDS++ Linker'
	cd $(BUILD_DIR) ; $(CC) -o $(EXECUTABLE) $(OBJECTS) $(EMU_FLAGS)
	@echo 'SDx Completed Building Target: $@'
	@echo ' '

$(pwd)/$(BUILD_DIR)/%.o: $(pwd)/$(SRC_DIR)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: SDS++ Compiler'
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) ; $(CC) $(CFLAGS) -o $(LFLAGS) $(HW_FLAGS)
	@echo 'Finished building: $<'
	@echo ' '
	
