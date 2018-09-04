#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>
#include "bnn.h"
#include "utils.h"
#ifdef _WIN32
#define TESTROUTE "e:/Computer/HLS/BNN/data/test_b.dat"
#define LABELROUTE "e:/Computer/HLS/BNN/data/label.dat"
#else
	#define TESTROUTE "data/test_b.dat"
	#define LABELROUTE "data/label.dat"
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

int main(int argc, char** argv){

	int8_t** test_images;
	test_images = new int8_t*[TEST_SIZE];
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[i] = new int8_t[784];
	int test_labels[TEST_SIZE];
	read_test_images(test_images);
	read_test_labels(test_labels);

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

	rosetta::CLMemObj inputMem((void*)input_image, sizeof(bit8_t), 784, CL_MEM_READ_ONLY);
	rosetta::CLMemObj outMem((void*)out, sizeof(fixo), OUT, CL_MEM_WRITE_ONLY);

	int global_size[3] = {1, 1, 1};
	int local_size[3] = {1, 1, 1};
	Bnn.set_global(global_size);
	Bnn.set_local(local_size);

	bnn_world.addKernel(Bnn);

	for (int test = 0; test < TEST_SIZE; test++) {

		for (int i = 0; i < 784; i++){
			input_image[i] = test_images[test][i];
		}
		#ifndef _WIN32
			timer_.start();
		#endif

		bnn_world.addMemObj(inputMem);
		bnn_world.addMemObj(outMem);

		bnn_world.setMemKernelArg(0, 0, 2*test);
		bnn_world.setMemKernelArg(0, 1, 2*test+1);

		bnn_world.runKernels(false);

		bnn_world.readMemObj(2*test+1);

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
