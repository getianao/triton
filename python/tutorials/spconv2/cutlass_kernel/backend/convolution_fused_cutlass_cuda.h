#pragma once

#include <torch/torch.h>

at::Tensor conv_forward_fused_cutlass_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer,
    std::vector<uint> cuda_streams);

at::Tensor conv_forward_fused_cutlass_subgroupedmm_fakeColumn_cuda(
    at::Tensor in_feat, at::Tensor kernel, const int kernel_volume,
    const int input_channels, const int output_channels,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, const int output_size,
    const bool GatherA, const bool ScatterD, const bool AtomicD,
    const bool transpose, const int config_no, const bool isHalf,
    at::Tensor mminfo, at::Tensor mminfo_cpu);

at::Tensor conv_forward_fused_cutlass_groupedmm_cuda(
    at::Tensor in_feat, at::Tensor kernel, const int kernel_volume,
    const int input_channels, const int output_channels,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, const int output_size,
    const bool GatherA, const bool ScatterD, const bool AtomicD,
    const bool transpose, const int config_no, const bool isHalf,
    at::Tensor mminfo, at::Tensor mminfo_cpu);

at::Tensor conv_forward_fused_cutlass_mm_cuda(
    at::Tensor in_feat, at::Tensor kernel, const int kernel_volume,
    const int input_channels, const int output_channels,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, const int output_size,
    const bool GatherA, const bool ScatterD, const bool AtomicD,
    std::vector<uint> cuda_streams, const bool transpose, const int config_no,
    const bool isHalf);