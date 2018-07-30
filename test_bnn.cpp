#include <iostream>
#include <fstream>
#include <cstdlib>
#include "bnn.h"
#include "model_dense.h"
#ifdef _WIN32
#define TESTROUTE "e:/Computer/HLS/BNN/data/test_b.dat"
#define LABELROUTE "e:/Computer/HLS/BNN/data/label.dat"
#else
#define TESTROUTE "/home/xl722/BNN/data/test_b.dat"
#define LABELROUTE "/home/xl722/BNN/data/label.dat"
#endif // WIN32


using namespace std;
const int TEST_SIZE = 1000;

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


int main(){

	int8_t** test_images;
	test_images = new int8_t*[TEST_SIZE];
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[i] = new int8_t[784];
	int test_labels[TEST_SIZE];
	read_test_images(test_images);
	read_test_labels(test_labels);

	float correct = 0.0;
	
	for (int test = 0; test < TEST_SIZE; test++) {

		bit input_image[I_WIDTH1*I_WIDTH1];
		bit output_image[O_WIDTH*O_WIDTH * 64] = { 0 };
		float output_image_f[O_WIDTH*O_WIDTH * 64] = { 0 };
		float layer1_out[512] = { 0 };
		float reshape_image[3136] = {0};
		float out[10] = { 0 };

		for (int i = 0; i < 784; i++)
			input_image[i] = test_images[test][i];

		bnn(input_image, output_image);
		for (int i = 0; i < O_WIDTH*O_WIDTH * 64; i++) output_image_f[i] = -output_image[i].to_int();
		reshape(output_image_f, reshape_image);
		dense(reshape_image, layer1_out, w_fc1, b_fc1, O_WIDTH*O_WIDTH*64, 512, true);
		dense(layer1_out, out, w_fc2, b_fc2, 512, 10, false);

		int max_id = 0;
		for(int i = 1; i < 10; i++)
			if(out[i] > out[max_id])
				max_id = i;
		if (max_id == test_labels[test]) correct += 1.0;
		cout << test << ": " << max_id << " " << test_labels[test] << endl;

	}
	cout << correct/TEST_SIZE << endl;
	
	return 0;
}
