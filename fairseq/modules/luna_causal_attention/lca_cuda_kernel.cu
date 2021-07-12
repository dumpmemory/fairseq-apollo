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
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ z,
    scalar_t* __restrict__ f) {


}

torch::Tensor lca_cuda_forward(
    torch::Tensor const& x,
    torch::Tensor const& y,
    torch::Tensor const& z) {
    /*
     Args:
         x: [len, bsz, dim1]
         y: [len, bsz, dim1]
         z: [len, bsz, dim2]
     return:
         f: [len, bsz, dim2]
    */
    
    const int len = x.size(0);
    const int bsz = x.size(1);
    const int dim_xy = x.size(2);
    const int dim_z = z.size(2);

    const int xy_inc_t = bsz * dim_xy;
    const int z_inc_t = bsz * dim_z;

    auto act_options  = x.options().requires_grad(false);
    torch::Tensor f = torch::zeros({len, bsz, dim_z}, act_options);

    
}