# BNN
This repository has the kernel code for a highly optimized BNN accelerator.
## file organization
The source codes are in directory \src, which contains 3 sub directories for 3 platforms.<br>
The weights and test data are in directory \data. The meaning of each file is explained as follow.<br>
* label.dat: labels for test data
* test_b.dat: test images
* weight_conv1: weights for convolution layer 1
* weight_conv2: weights for convolution layer 2
* weight_k1, weight_h1: batch_norm parameters for conv layer 1
* weight_k2, weight_h2: batch_norm parameters for conv layer 2
* weight_fc1_bp: weights for dense layer 1(for on-chip implementation)
* weight_fc2_bp: weights for dense layer 2(for on-chip implementation)
* weight_10bp: weights for dense layer 1(for off-chip implementation)
* weight_12bp: weights for dense layer 2(for off-chip implementation)
* weight_bfc1: bias for dense layer 1
* weight_bfc2: bias for dense layer 2
## platform details
There are three implementations for BNN: Zedboard, zc706 and AWS F1.
### Zedboard
The scale of this accelerator is the smallest among these three. The parallelism of convolution layer 1 is 32(32 output fmaps being processed at the same time). The parallelism of convolution layer 2 is 16. No line buffer or dataflow is used in this design.
### zc706
The parallelism of conv 1 is 32, while that of conv 2 is 8. Line buffer is ued in convolution layer and dataflow is used in dense layer.
### AWS F1
Every conv layer is fully parallized which means the parallelism of conv 1 and that of conv 2 is 64. Line buffer and dataflow are used. Weights for dense layer are now stored on chip to improve the speed of data transfer.
