PLATFORM := zc706

BUILD_DIR := build/$(PLATFORM)_$(TARGET_OS)_$(TARGET)

pwd := $(CURDIR)

APPSOURCES := test_bnn.cpp layer.cpp bnn.cpp

EXCUTABLE := bnn.elf

SDSFLAGS := -sds-pf $(PLATFORM) \
	-sds-hw 
