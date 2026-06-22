#include <torch/extension.h>

torch::Tensor layernorm_cuda(torch::Tensor x);

torch::Tensor layernorm(torch::Tensor x) {
    return layernorm_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &layernorm, "LayerNorm forward (CUDA)");
}