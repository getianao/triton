#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution_fused_cutlass_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_forward_fused_cutlass_groupedmm_cuda",
        &conv_forward_fused_cutlass_groupedmm_cuda);
}
