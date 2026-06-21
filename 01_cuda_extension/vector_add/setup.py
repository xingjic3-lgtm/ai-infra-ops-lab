from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="vector_add_ext",
    ext_modules=[
        CUDAExtension(
            name="vector_add_ext",
            sources=["binding.cpp", "vector_add_cuda.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)