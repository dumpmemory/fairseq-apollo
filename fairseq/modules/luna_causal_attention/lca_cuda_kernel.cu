#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <stdio.h>

#define FULL_MASK 0xffffffff


template<typename scalar_t>
__global__ 
void lca_cuda_forward_kernel(
    )
