#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <time.h>
#include <sys/time.h>

#include <cstdlib>
// not necessary? #include "BNN.h"  //changed
#include "model_dense.h"
#include "utils.h"

#ifdef OCL
// opencl harness headers
#include "CLWorld.h"
#include "CLKernel.h"
#include "CLMemObj.h"
// harness namespace
using namespace rosetta;
#endif

//using namespace std;  !!!!
const int TEST_SIZE = 1000;

void read_test_images(int8_t** test_images) {
  std::ifstream infile("data/test_b.dat");
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
  std::ifstream infile("data/label.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      infile >> test_labels[index];
    }
    infile.close();
  }
}


int main(int argc, char ** argv){
    
    std::cout<<"BNN Application\n";
    
#ifdef OCL
    // parse command line arguments for opencl version
    std::string kernelFile("");
    //parse_sdaccel_command_line_args(argc, argv, kernelFile);
#endif
    
    int8_t** test_images;
    test_images = new int8_t*[TEST_SIZE];
    for(int i = 0; i < TEST_SIZE; i++)
        test_images[i] = new int8_t[784];
    int test_labels[TEST_SIZE];
    read_test_images(test_images);
    read_test_labels(test_labels);
    
    float correct = 0.0;
    
    for (int test = 0; test < TEST_SIZE; test++) {
        
        bit input_image[I_WIDTH1*I_WIDTH1];             //may need to convert to bytes
        bit output_image[O_WIDTH*O_WIDTH * 64] = { 0 };
        float output_image_f[O_WIDTH*O_WIDTH * 64] = { 0 };
        float layer1_out[512] = { 0 };
        float reshape_image[3136] = {0};
        float out[10] = { 0 };
        
        for (int i = 0; i < 784; i++)
            input_image[i] = test_images[test][i];
  
        
        
        
        
    struct timeval start, end;
    
    // opencl version host code
#ifdef OCL
    // create OpenCL world
    CLWorld bnn_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);
    
    // add the bitstream file
    bnn_world.addProgram(kernelFile);
    
    // create kernels
    CLKernel BNN(bnn_world.getContext(), bnn_world.getProgram(), "BNN", bnn_world.getDevice());
    
    // create mem objects
    CLMemObj input_mem ( (void*)input_image,  sizeof(bit), I_WIDTH1*I_WIDTH1,     CL_MEM_READ_ONLY);
    CLMemObj output_mem  ( (void*)output_image,   sizeof(bit), O_WIDTH*O_WIDTH * 64,        CL_MEM_WRITE_ONLY);
    
    // start timer
    gettimeofday(&start, 0);
    
    // add them to the world
    // added in sequence, each of them can be referenced by an index
    bnn_world.addMemObj(input_mem);
    bnn_world.addMemObj(output_mem);
    
    // set work size
    int global_size[2] = {1, 1};
    int local_size[2] = {1, 1};
    BNN.set_global(global_size);
    BNN.set_local(local_size);
    
    // add them to the world
    bnn_world.addKernel(BNN);
    
    // set kernel arguments
    bnn_world.setMemKernelArg(0, 0, 0);  //not sure about the numbers!!
    bnn_world.setMemKernelArg(0, 1, 1);
    
    // run!
    bnn_world.runKernels();
    
    // read the data back
    bnn_world.readMemObj(1);
    
    // end timer
    gettimeofday(&end, 0);
        
    bnn_world.releaseWorld();

//#else
  
    //BNN(input_image, output_image);   //kernel here?


#endif
    //reconvert fixed to float not necessary? output in bits
        
    for (int i = 0; i < O_WIDTH*O_WIDTH * 64; i++) output_image_f[i] = -output_image[i].to_int(); 
        //delete []output_image;
        reshape(output_image_f, reshape_image);
    dense(reshape_image, layer1_out, w_fc1, b_fc1, O_WIDTH*O_WIDTH*64, 512, true);
    dense(layer1_out, out, w_fc2, b_fc2, 512, 10, false);

    int max_id = 0;
    for(int i = 1; i < 10; i++)
      if(out[i] > out[max_id])
        max_id = i;
    if (max_id == test_labels[test]) correct += 1.0;
      std::cout << test << ": " << max_id << " " << test_labels[test] << std::endl;

  }
    std::cout << correct/TEST_SIZE << std::endl;
    

  
	return 0;
}
