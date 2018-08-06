PLATFORM := zc706

BUILD_DIR := build
SRC := src

pwd := $(CURDIR)

APPSOURCES := test_bnn.cpp layer.cpp bnn.cpp

EXCUTABLE := bnn.elf

SDSFLAGS := -sds-pf $(PLATFORM) \
	-sds-hw bnn $(SRC)/bnn.cpp -sds-end \
	-poll-mode 1

CC := sds++ $(SDSFLAGS)

CFLAGS = -Wall -c
LFALGS = 

OBJECTS := $(APPSOURCES:.cpp=.o)
SOURCES := $(patsubst %, $(SRC)/%, $(APPSOURCES))

.PHONY: all
all: ${BUILD_DIR}/${EXCUTABLE}

${BUILD_DIR}/${EXCUTABLE}: ${OBJECTS}
	mkdir -p ${BUILD_DIR}
	${CC} ${LFALGS} $< -o $@

%.o: ${SOURCES}
	${CC} ${CFLAGS} $<
