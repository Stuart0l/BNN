#include <iostream>
#include <fstream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include "bnn.h"
#include "utils.h"
#ifdef __SDSCC__
	#include "sds_lib.h"
#endif
#ifdef _WIN32
#define TESTROUTE "e:/Computer/HLS/BNN/data/test_b.dat"
#define LABELROUTE "e:/Computer/HLS/BNN/data/label.dat"
#define WFC1ROUTE "e:/Computer/HLS/BNN/data/weight_10bp"
#define WFC2ROUTE "e:/Computer/HLS/BNN/data/weight_12bp"
#else
	#define TESTROUTE "data/test_b.dat"
	#define LABELROUTE "data/label.dat"
	#define WFC1ROUTE "data/weight_10bp"
	#define WFC2ROUTE "data/weight_12bp"
	#include "timer.h"
#endif
#include "CLWorld.h"

const int TEST_SIZE = 500;

void read_test_images(int8_t** test_images) {
	std::ifstream infile(TESTROUTE);
	if (infile.is_open()) {
		for (int index = 0; index < TEST_SIZE; index++) {
			for (int pixel = 0; pixel < 784; pixel++) {
				int i;
				infile >> i;
				test_images[index][pixel] = i;
			}
		}
		infile.close();
	}
}

void read_test_labels(int test_labels[TEST_SIZE]) {
	std::ifstream infile(LABELROUTE);
	if (infile.is_open()) {
		for (int index = 0; index < TEST_SIZE; index++) {
			infile >> test_labels[index];
		}
		infile.close();
	}
}

void read_fc1_weights(bit64_t* w_fc1) {
	std::ifstream infile(WFC1ROUTE);
	if (infile.is_open()) {
		for (int index = 0; index < MAX_W_FC / 64; index++) {
			unsigned long long i;
			infile >> i;
			w_fc1[index] = i;
		}
		infile.close();
	}
}

void read_fc2_weights(bit64_t* w_fc2) {
	std::ifstream infile(WFC2ROUTE);
	if (infile.is_open()) {
		for (int index = 0; index < 80; index++) {
			unsigned long long i;
			infile >> i;
			w_fc2[index] = i;
		}
		infile.close();
	}
}

int main(int argc, char** argv){

	int8_t** test_images;
	#ifdef __SDSCC__
		bit64_t* w_fc1 = (bit64_t*)sds_alloc(MAX_W_FC / 64 * sizeof(bit64_t));
		bit64_t* w_fc2 = (bit64_t*)sds_alloc(80 * sizeof(bit64_t));
	#else
		bit64_t* w_fc1 = new bit64_t[MAX_W_FC / 64];
		bit64_t* w_fc2 = new bit64_t[80];
	#endif
	test_images = new int8_t*[TEST_SIZE];
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[i] = new int8_t[784];
	int test_labels[TEST_SIZE];
	read_test_images(test_images);
	read_test_labels(test_labels);
	read_fc1_weights(w_fc1);
	read_fc2_weights(w_fc2);

	float correct = 0.0;

	bit8_t input_image[I_WIDTH1*I_WIDTH1];
	fixo out[OUT];
	float result[OUT];
	#ifndef _WIN32
		Timer timer_("bnn timer");
	#endif

	rosetta::CLWorld bnn_world(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);

	std::string KernelFile;
	parse_sdaccel_command_line_args(argc, argv, KernelFile);
	bnn_world.addProgram(KernelFile);

	rosetta::CLKernel Bnn(bnn_world.getContext(), bnn_world.getProgram(), "bnn", bnn_world.getDevice());
	bnn_world.addKernel(Bnn);

	rosetta::CLMemObj weightfc1((void*)w_fc1, sizeof(bit64_t), MAX_W_FC / 64, CL_MEM_READ_ONLY);
	rosetta::CLMemObj weightfc2((void*)w_fc2, sizeof(bit64_t), 80, CL_MEM_READ_ONLY);
	rosetta::CLMemObj inputMem((void*)input_image, sizeof(bit8_t), 784, CL_MEM_READ_ONLY);
	rosetta::CLMemObj outMem((void*)out, sizeof(fixo), OUT, CL_MEM_WRITE_ONLY);

	int global_size[3] = {1, 1, 1};
	int local_size[3] = {1, 1, 1};
	Bnn.set_global(global_size);
	Bnn.set_local(local_size);

	bnn_world.addMemObj(weightfc1);
	bnn_world.addMemObj(weightfc2);

	for (int test = 0; test < TEST_SIZE; test++) {

		for (int i = 0; i < 784; i++){
			input_image[i] = test_images[test][i];
		}
		#ifndef _WIN32
			timer_.start();
		#endif

		bnn_world.addMemObj(inputMem);
		bnn_world.addMemObj(outMem);

		bnn_world.setMemKernelArg(0, 0, 2*test+2);
		bnn_world.setMemKernelArg(0, 1, 2*test+3);
		bnn_world.setMemKernelArg(0, 2, 0);
		bnn_world.setMemKernelArg(0, 3, 1);

		bnn_world.runKernels();

		bnn_world.readMemObj(2*test+3);

		for (int i = 0; i < 10; i++)
			result[i] = out[i].to_float();

		int max_id = 0;
		for (int i = 1; i < 10; i++) {
			if (result[i] > result[max_id])
				max_id = i;
		}
		#ifndef _WIN32
			timer_.stop();
		#endif
		if (max_id == test_labels[test]) correct += 1.0;
		std::cout << test << ": " << max_id << " " << test_labels[test] << std::endl;
	}
	std::cout << correct/TEST_SIZE << std::endl;
	bnn_world.releaseWorld();

	return 0;
}
