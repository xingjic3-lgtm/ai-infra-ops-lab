ai-infra-ops-lab/
├── 01_cuda_extension/
│   ├── vector_add/
│   ├── layernorm/
│   └── matmul/
├── 02_triton_kernels/
│   ├── layernorm/
│   ├── matmul/
│   └── attention/
├── 03_tensorrt/
│   ├── export_onnx/
│   ├── build_engine/
│   ├── run_engine/
│   └── plugin/
├── 04_kv_cache/
│   ├── contiguous_kv_cache/
│   ├── paged_kv_cache/
│   └── paged_attention_kernel/
├── 05_vllm_hacking/
│   ├── scheduler_log/
│   ├── block_allocation_log/
│   └── config_experiments/
└── 06_distributed/
    ├── all_reduce/
    ├── ddp_toy/
    ├── tensor_parallel_linear/
    └── pipeline_parallel_toy/