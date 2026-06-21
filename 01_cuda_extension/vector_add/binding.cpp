#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    return vector_add_cuda(a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector add CUDA");
}