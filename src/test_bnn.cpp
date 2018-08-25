#include <iostream>
#include <fstream>
#include <cstdlib>
#include "bnn.h"
#include "timer.h"
#ifdef __SDSCC__
#include "sds_lib.h"
#endif
#ifdef _WIN32
#define TESTROUTE "e:/Computer/HLS/BNN/data/test_b.dat"
#define LABELROUTE "e:/Computer/HLS/BNN/data/label.dat"
#else
#define TESTROUTE "data/test_b.dat"
#define LABELROUTE "data/label.dat"
#endif // WIN32

using namespace std;
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
	std::ifstream infile("data/weight_10bp");
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
	std::ifstream infile("data/weight_12bp");
	if (infile.is_open()) {
		for (int index = 0; index < 80; index++) {
			unsigned long long i;
			infile >> i;
			w_fc2[index] = i;
		}
		infile.close();
	}
}

int main(){

	int8_t** test_images;
	#ifdef __SDSCC__
	bit64_t* w_fc1 = (bit64_t*)sds_alloc(MAX_W_FC / 64 * sizeof(bit8_t));
	bit64_t* w_fc2 = (bit64_t*)sds_alloc(80 * sizeof(bit8_t));
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
	fixo out[10];
	float result[10];

	Timer timer("conv timer");
	Timer timer_("bnn timer");

	for (int test = 0; test < TEST_SIZE; test++) {

		for (int i = 0; i < 784; i++){
			input_image[i] = test_images[test][i];
		}

		timer_.start();
		timer.start();

		bnn(input_image, out, w_fc1, w_fc2);

		for (int i = 0; i < 10; i++)
			result[i] = out[i].to_float();

		timer.stop();

		int max_id = 0;
		for (int i = 1; i < 10; i++) {
			if (result[i] > result[max_id])
				max_id = i;
		}
		timer_.stop();
		if (max_id == test_labels[test]) correct += 1.0;
		cout << test << ": " << max_id << " " << test_labels[test] << endl;
	}
	cout << correct/TEST_SIZE << endl;

	return 0;
}
