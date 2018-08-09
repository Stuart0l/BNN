#include <iostream>
#include <stdint.h>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include "typedef.h"
#include "model_dense.h"
#include "utils.h"

#ifdef OCL
// opencl harness headers
#include "CLWorld.h"
#include "CLKernel.h"
#include "CLMemObj.h"
// harness namespace
// using namespace rosetta;
#endif

const int TEST_SIZE = 80;

void read_test_images(bit8_t** test_images) {
   using namespace std;
  
std::ifstream infile1("data/test_b.dat");
  if (infile1.is_open()) {
	bit8_t i; 
  for (int index = 0; index < TEST_SIZE; index++) {
      for (int pixel = 0; pixel < 784; pixel++) {
        infile1 >> i;
        test_images[index][pixel] = i;
      }
  }
    infile1.close();
  }
}

void read_test_labels(int test_labels[TEST_SIZE]) {
using namespace std;  
std::ifstream infile2("data/label.dat");
  if (infile2.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      infile2 >> test_labels[index];
    }
    infile2.close();
  }
}


int main(int argc, char ** argv){
    
    std::cout<<"BNN Application\n";
    
#ifdef OCL
    // parse command line arguments for opencl version
    std::string kernelFile("");
    parse_sdaccel_command_line_args(argc, argv, kernelFile);
#endif
    
    bit8_t** test_images;
    test_images = new bit8_t*[TEST_SIZE];
    for(int i = 0; i < TEST_SIZE; i++)
        test_images[i] = new bit8_t[784];
    int test_labels[TEST_SIZE];
    read_test_images(test_images);
    read_test_labels(test_labels);
    
    float correct = 0.0;
    long long total_elapsed=0;

    for (int test = 0; test<TEST_SIZE; test++) {
        
        bit8_t io_image[O_WIDTH*O_WIDTH * 64] = { 0 };
        float output_image_f[O_WIDTH*O_WIDTH * 64] = { 0 };
        float layer1_out[512] = { 0 };
        float reshape_image[3136] = {0};
        float out[10] = { 0 };
        
        for (int i = 0; i < 784; i++)
            io_image[i] = test_images[test][i];
    
        
        struct timeval start, end;
        
        // opencl version host code
    #ifdef OCL
        using namespace rosetta;
        // create OpenCL world
        CLWorld bnn_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);
        
        // add the bitstream file
        bnn_world.addProgram(kernelFile);
        
        // create kernels
        CLKernel BNN(bnn_world.getContext(), bnn_world.getProgram(), "BNN", bnn_world.getDevice());
        
        // create mem objects
        CLMemObj input_mem ( (void*)io_image,  sizeof(bit8_t), O_WIDTH*O_WIDTH * 64,   CL_MEM_READ_WRITE);
        
        // start timer
        gettimeofday(&start, 0);
        
        // add them to the world
        // added in sequence, each of them can be referenced by an index
        bnn_world.addMemObj(input_mem);
        
        // set work size
        int global_size[3] = {1, 1, 1};
        int local_size[3] = {1, 1, 1};
        BNN.set_global(global_size);
        BNN.set_local(local_size);
        
        // add them to the world
        bnn_world.addKernel(BNN);
        
        // set kernel arguments
        bnn_world.setMemKernelArg(0, 0, 0);
        
        // run!
        bnn_world.runKernels();
        
        // read the data back
        bnn_world.readMemObj(0);
        
        // end timer
        gettimeofday(&end, 0);
        
        bnn_world.releaseWorld();

    #else
      
        BNN(io_image);

    #endif
        
    // print time
    long long elapsed = (end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec;
    total_elapsed += elapsed/TEST_SIZE;
    std::cout<<"elapsed time: "<<elapsed<<" us\n";
        
    for (int i = 0; i < O_WIDTH*O_WIDTH * 64; i++)
        output_image_f[i] = -io_image[i];
	
	reshape(output_image_f, reshape_image);
    dense(reshape_image, layer1_out, w_fc1, b_fc1, O_WIDTH*O_WIDTH*64, 512, true);
    dense(layer1_out, out, w_fc2, b_fc2, 512, 10, false);

    int max_id = 0;
    for(int i = 1; i < 10; i++)
      if(out[i] > out[max_id])
        max_id = i;
    if (max_id == test_labels[test]) correct += 1.0;
      std::cout <<"TEST "<< test << " result: " << max_id << " expected: " << test_labels[test] << std::endl;

}
    
    std::cout <<"average time of operation: "<<total_elapsed<<std::endl;

    std::cout <<"correct: "<< correct << " test size: " << TEST_SIZE << " correct percentage: " << correct/TEST_SIZE << std::endl;
        //delete []output_image;


  
	return 0;
}
