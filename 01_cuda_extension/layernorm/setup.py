from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="layernorm_ext",
    ext_modules=[
        CUDAExtension(
            name="layernorm_ext",
            sources=[
                "layernorm.cpp",
                "layernorm_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)