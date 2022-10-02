#ifndef PCRY_CUH
#define PCRY_CUH

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK 256

// DEFINE SOME CONSTANTS IN DEVICE
__constant__ float BOLTZMANN   = 1.380e-23;
__constant__ float ION_MASS    = 6.680e-26;
__constant__ float PI          = 3.1415926;

#endif
