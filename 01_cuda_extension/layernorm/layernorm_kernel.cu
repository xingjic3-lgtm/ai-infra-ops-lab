#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_kernel( const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float shared_sum[256];
    __shared__ float shared_var[256];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += block_size) {
        local_sum += x[row * cols + col];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared_sum[0] / cols;

    float local_var = 0.0f;
    for (int col = tid; col < cols; col += block_size) {
        float diff = x[row * cols + col] - mean;
        local_var += diff * diff;
    }

    shared_var[tid] = local_var;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }

    float var = shared_var[0] / cols;

    for (int col = tid; col < cols; col += block_size) {
        y[row * cols + col] = (x[row * cols + col] - mean) / sqrtf(var + 1e-5f);
    }
}

torch::Tensor layernorm_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);

    int rows = x.size(0);
    int cols = x.size(1);

    layernorm_kernel<<<rows, 256>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        cols
    );

    return y;
}