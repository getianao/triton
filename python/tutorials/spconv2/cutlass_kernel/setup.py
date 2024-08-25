import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from version import __version__

cutlass_dir = "/home/tge/workspace/triton/python/tutorials/spconv2/cutlass_kernel/cutlass-3dconv"
if not os.path.isdir(cutlass_dir):
    raise Exception(
        "Environment variable CUTLASS_DIR must point to the CUTLASS installation"
    )


sources = [
    os.path.join("backend", f"pybind_cuda.cpp"),
    os.path.join("backend", f"convolution_fused_cutlass_cuda.cu"),
]

_cutlass_include_dirs = ["tools/util/include", "include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

# Set additional flags needed for compilation here
cxx_flags = ["-O3", "-fopenmp", "-lgomp", "-std=c++17"] # "-g"
nvcc_flags = ["-O3", "-lineinfo", "-DNDEBUG", "-std=c++17", "-arch=sm_86"]
ld_flags = []

# For the hopper example, we need to specify the architecture. It also needs to be linked to CUDA library.
if "COMPILE_3X_HOPPER" in os.environ:
    nvcc_flags += [
        "--generate-code=arch=compute_90a,code=[sm_90a]",
        "-DCOMPILE_3X_HOPPER",
    ]
    ld_flags += ["cuda"]

setup(
    name="cutlass_spconv",
    version=__version__,
    ext_modules=[
        CUDAExtension(
            name="cutlass_spconv",
            sources=sources,
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            libraries=ld_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
