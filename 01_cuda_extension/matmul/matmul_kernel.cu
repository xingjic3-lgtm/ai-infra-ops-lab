#include <torch/extension.h>

__global__ void matmul_kernel(const float* a, const float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

torch::Tensor matmul_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a.size(1) must equal b.size(0)");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());
 
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M,
        K,
        N
    );

    return c;
}