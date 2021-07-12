/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <vector>


torch::Tensor lca_cuda_forward(
    torch::Tensor const& x,
    torch::Tensor const& y,
    torch::Tensor const& z);

std::vector<torch::Tensor> lca_cuda_backward(
    torch::Tensor const& x,
    torch::Tensor const& y,
    torch::Tensor const& z);


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)

torch::Tensor lca_forward(
    torch::Tensor const& x,
    torch::Tensor const& y,
    torch::Tensor const& z) {

    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(z);

    return lca_cuda_forward(x, y, z);
}

std::vector<torch::Tensor> lca_backward(
    torch::Tensor const& x,
    torch::Tensor const& y,
    torch::Tensor const& z) {

    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(z);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lca_forward, "luna causal forward (CUDA)");
    m.def("backward", &lca_backward, "luna causal backward (CUDA)");
}
