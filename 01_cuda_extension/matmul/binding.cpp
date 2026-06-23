#include <torch/extension.h>

torch::Tensor matmul_forward(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Matmul forward CUDA");
}