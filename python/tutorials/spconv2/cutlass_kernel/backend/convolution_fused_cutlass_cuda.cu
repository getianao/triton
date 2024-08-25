#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>

#include "convolution_fused_cutlass_cuda.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped_gather_scatter.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped_gather_scatter.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/thread/linear_combination.h"

at::Tensor conv_forward_fused_cutlass_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor global_buffer,
    std::vector<uint> cuda_streams) {

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.
  using ElementAccumulator = float; // <- data type of accumulator
  using ElementComputeEpilogue =
      ElementAccumulator; // <- data type of epilogue operations
  using ElementInputA =
      cutlass::half_t; // <- data type of elements in input matrix A
  using ElementInputB =
      cutlass::half_t;         // <- data type of elements in input matrix B
  using ElementOutput = float; // <- data type of elements in output matrix D

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, B and C.
  //
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 128, 16>; // <- threadblock tile M = 128, N
                                              // = 128, K = 16
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<64, 64,
                               16>; // <- warp tile M = 64, N = 64, K = 16
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8
  // 16, 8, 8 -> Turing
  // 16, 8, 16 -> Ampere

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha in linear combination
                               // function

  // Number of pipelines you want to use
  constexpr int NumStages = 5;
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages,
      4, /*alignmentA*/
      4, /*alignmentB*/
      cutlass::arch::OpMultiplyAdd, cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone, true, /*GatherA*/
      false,                                  /*GatherB*/
      true,                                    /*ScatterD*/
      cutlass::layout::NoPermute,
      cutlass::layout::NoPermute, cutlass::layout::NoPermute, true /*AtomicD*/
      >;

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(in_feat.device());
  at::Tensor out_feat = torch::zeros({output_size, kernel.size(-1)}, options);

  int in_nnz = in_feat.size(0);
  int out_nnz = out_feat.size(0);
  int in_channel = in_feat.size(1);
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);

  // float *in_feat_ptr = in_feat.data_ptr<float>();
  // float *weight_ptr = kernel.data_ptr<float>();
  half *in_feat_ptr = reinterpret_cast<half *>(in_feat.data_ptr<at::Half>());
  half *weight_ptr = reinterpret_cast<half *>(kernel.data_ptr<at::Half>());
  float *out_feat_ptr = out_feat.data_ptr<float>();
  int *in_map_ptr = neighbor_map.data_ptr<int>(); // todo
  int *out_map_ptr = neighbor_map.data_ptr<int>();

  // int *kpos_ptr = kernel_pos.data_ptr<int>();

  //   cutlass fused gather-mm-scatter kernel
  // cutlass::gemm::GemmCoord problem_size(in_nnz, in_channel, out_channel);

  // Initialize alpha/beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  // loop over all kernel offsets
  int cur_idx = 0;
  // int stream_id = 0;

  // auto start_time = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < k_vol; k++) {
    int cur_nnz = neighbor_offset.data_ptr<int>()[k];
    if (cur_nnz == 0) {
      continue;
    }
    int cuda_stream_id = k % cuda_streams.size();
    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(cuda_streams[cuda_stream_id]);

    cutlass::gemm::GemmCoord problem_size_real(cur_nnz, out_channel,
                                               in_channel);
    // Create a tuple of gemm kernel arguments. This is later passed as
    // arguments to launch instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size_real, // <- problem size of matrix multiplication
        split_k_slices,    // <- k-dimension split factor
        {alpha, beta},     // <- alpha, beta
        in_feat_ptr,       // <- reference to matrix A on device
        &weight_ptr[k * in_channel *
                    out_channel], // <- reference to matrix B on device
        out_feat_ptr,  // out_feat_ptr,    // <- reference to matrix C on device
        out_feat_ptr, // <- reference to matrix D on device
        cur_nnz * in_channel,     //   options.index_size * problem_size.k(),
        in_channel * out_channel, //   problem_size.k() * problem_size.n(),
        in_nnz * out_channel,     //   problem_size.m() * problem_size.n(),
        in_nnz * out_channel,     //   problem_size.m() * problem_size.n(),
        cutlass::layout::RowMajor::Stride(in_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        &in_map_ptr[cur_idx],
        nullptr, // <- pointer to index vector to gather  B on device
        &out_map_ptr[cur_idx],
    }; // <- pointer to index vector to scatter D on device

    // // Using the arguments, query for extra workspace required for matrix
    // // multiplication computation
    // size_t workspace_size = Gemm::get_workspace_size(arguments);
    // // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    // // Instantiate CUTLASS kernel depending on templates
    // Gemm gemm_op;
    // // Check the problem size is supported or not
    // cutlass::Status status = gemm_op.can_implement(arguments);
    // //   CUTLASS_CHECK(status);
    // // Initialize CUTLASS kernel with arguments and workspace pointer
    // status = gemm_op.initialize(arguments, workspace.get());
    // //   CUTLASS_CHECK(status);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(arguments);

    for (int i = 0; i < 1; i++) {
      // Launch initialized CUTLASS kernel
      status = gemm_op(cuda_stream);
      // status = gemm_op();
      //   CUTLASS_CHECK(status);
    }
  }

  return out_feat;
}

template <typename type_groupmm, typename datatype, typename torch_datatype>
at::Tensor launch_cutlass_subgroupedmm_fakeColumn_cuda(at::Tensor &in_feat,
                                                      at::Tensor &kernel,
                                                      const int kernel_volume,
                                                      const int input_channels,
                                                      const int output_channels,
                                                      at::Tensor &neighbor_map, // [2, 13*4+1, q]
                                                      at::Tensor &neighbor_offset, // CPU
                                                      const int output_size,
                                                      const bool transpose,
                                                      const bool AtomicD,
                                                      at::Tensor mminfo,
                                                      at::Tensor mminfo_cpu) {
  // TODO(tge): datatype
  using InputType = datatype;
  using OutputType = datatype;
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  at::Tensor out_feat = torch::zeros({output_size, kernel.size(-1)}, options);
  
  using LayoutA = typename type_groupmm::LayoutA;
  using LayoutB = typename type_groupmm::LayoutB;
  using LayoutC = typename type_groupmm::LayoutC;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<type_groupmm>;
  typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(OutputType(1),
                                                             OutputType(1));

  InputType *in_feat_ptr =
      reinterpret_cast<InputType *>(in_feat.data_ptr<torch_datatype>());
  InputType *weight_ptr =
      reinterpret_cast<InputType *>(kernel.data_ptr<torch_datatype>());
  OutputType *out_feat_ptr =
      reinterpret_cast<OutputType *>(out_feat.data_ptr<torch_datatype>());

  OutputType *C_ptr = nullptr;
  if (AtomicD == true) {
    C_ptr = nullptr;
  } else {
    C_ptr = out_feat_ptr;
  }

  int pqOffset = neighbor_map.size(1) * neighbor_map.size(2);
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose) {
    in_map_ptr = neighbor_map.data_ptr<int>() + pqOffset;
    out_map_ptr = neighbor_map.data_ptr<int>();
  } else {
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + pqOffset;
  }
  int *neighbor_offset_ptr = neighbor_offset.data_ptr<int>();

  // TODO(tge): only works for kernel_volume=27  
  if (kernel_volume % 2 == 1) {
    // 2 MMs to 3 MMs
    int subgroup_number =
        (kernel_volume + 1) / 2; // subgroupedMMs to be launched: 27 ->14
    int offset_num = (subgroup_number - 1) * 4 +
                     1; // all MMs in subgroupedMMs: 27 -> 13*4+1 = 53


    cutlass::gemm::GemmCoord *problem_sizes_device;
    int **in_feat_indices_device;
    int **out_feat_indices_device;
    int64_t *lda;
    int64_t *ldb;
    int64_t *ldd;
    InputType **weight_ptr_array_device;

    int align_test = 64;
    int global_mem_size =
        align_test * (2 * sizeof(int *) + sizeof(InputType *) +
                      sizeof(cutlass::gemm::GemmCoord) + 3 * sizeof(int64_t));
    in_feat_indices_device =
        reinterpret_cast<int **>(mminfo.data_ptr<uint8_t>());
    out_feat_indices_device = in_feat_indices_device + align_test;
    weight_ptr_array_device =
        reinterpret_cast<InputType **>(out_feat_indices_device + align_test);
    problem_sizes_device = reinterpret_cast<cutlass::gemm::GemmCoord *>(
        weight_ptr_array_device + align_test);
    lda = reinterpret_cast<int64_t *>(problem_sizes_device + align_test);
    ldb = lda + align_test;
    ldd = ldb + align_test;

    int **in_feat_indices =
        reinterpret_cast<int **>(mminfo_cpu.data_ptr<uint8_t>());
    int **out_feat_indices = in_feat_indices + align_test;
    InputType **weight_ptr_array =
        reinterpret_cast<InputType **>(out_feat_indices + align_test);
    cutlass::gemm::GemmCoord *problem_sizes =
        reinterpret_cast<cutlass::gemm::GemmCoord *>(weight_ptr_array +
                                                     align_test);
    int64_t *lda_host = reinterpret_cast<int64_t *>(problem_sizes + align_test);
    int64_t *ldb_host = lda_host + align_test;
    int64_t *ldd_host = ldb_host + align_test;
    int **in_feat_indices_p2 = reinterpret_cast<int **>(ldd_host + align_test);

    // auto start_time2 = std::chrono::high_resolution_clock::now();
    int nz_num = 0;
    const int nbmaps_alignment = neighbor_map.size(2);
    for (int subgroup_id = 0; subgroup_id < subgroup_number - 1;
         subgroup_id++) { // exclude the last one, 14-1=13
      //  TODO(tge): empty MM
      
      int real_subgroup_id = 4 * subgroup_id;
      int problem_size_0 = neighbor_offset_ptr[real_subgroup_id];
      int problem_size_1 = neighbor_offset_ptr[real_subgroup_id + 1];
      int problem_size_2 = neighbor_offset_ptr[real_subgroup_id + 2];
      problem_sizes[3 * nz_num] = cutlass::gemm::GemmCoord(
          problem_size_0, output_channels, input_channels); // mnk
      problem_sizes[3 * nz_num + 1] = cutlass::gemm::GemmCoord(
          problem_size_1, output_channels, input_channels);
      problem_sizes[3 * nz_num + 2] =
          cutlass::gemm::GemmCoord(problem_size_2, output_channels,
                                   input_channels * 2); // fakeColumn here
      weight_ptr_array[3 * nz_num] =
          &weight_ptr[2 * subgroup_id * input_channels * output_channels];
      weight_ptr_array[3 * nz_num + 1] =
          &weight_ptr[(2 * subgroup_id + 1) * input_channels * output_channels];
      weight_ptr_array[3 * nz_num + 2] =
          &weight_ptr[2 * subgroup_id * input_channels * output_channels];

      int offset_0 = real_subgroup_id * nbmaps_alignment;
      in_feat_indices[3 * nz_num] = &in_map_ptr[offset_0];
      out_feat_indices[3 * nz_num] = &out_map_ptr[offset_0];
      int offset_1 = (real_subgroup_id + 1) * nbmaps_alignment;
      in_feat_indices[3 * nz_num + 1] = &in_map_ptr[offset_1];
      out_feat_indices[3 * nz_num + 1] = &out_map_ptr[offset_1];
      int offset_2 = (real_subgroup_id + 2) * nbmaps_alignment;
      in_feat_indices[3 * nz_num + 2] = &in_map_ptr[offset_2];
      out_feat_indices[3 * nz_num + 2] = &out_map_ptr[offset_2];
      int offset_3 = (real_subgroup_id + 3) * nbmaps_alignment;
      in_feat_indices_p2[nz_num] = &in_map_ptr[offset_3];

      for (int i = 0; i < 3; i++) {
        auto problem = problem_sizes[3 * nz_num + i];
        lda_host[3 * nz_num + i] =
            LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host[3 * nz_num + i] =
            LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldd_host[3 * nz_num + i] =
            LayoutC::packed({problem.m(), problem.n()}).stride(0);
      }
      nz_num++;
    }
    // last one
    int problem_size_0 = neighbor_offset_ptr[4 * (subgroup_number - 1)];
    problem_sizes[3 * nz_num] = cutlass::gemm::GemmCoord(
        problem_size_0, output_channels, input_channels); // mnk
    weight_ptr_array[3 * nz_num] =
        &weight_ptr[2 * (subgroup_number - 1) * input_channels *
                    output_channels];
    int offset_0 = 4 * (subgroup_number - 1) * nbmaps_alignment;
    in_feat_indices[3 * nz_num] = &in_map_ptr[offset_0];
    out_feat_indices[3 * nz_num] = &out_map_ptr[offset_0];
    lda_host[3 * nz_num] =
        LayoutA::packed({problem_size_0, input_channels}).stride(0);
    ldb_host[3 * nz_num] =
        LayoutB::packed({input_channels, output_channels}).stride(0);
    ldd_host[3 * nz_num] =
        LayoutC::packed({problem_size_0, output_channels}).stride(0);

    cudaMemcpy(in_feat_indices_device, in_feat_indices, global_mem_size,
               cudaMemcpyHostToDevice);

    // auto end_time2 = std::chrono::high_resolution_clock::now();
    // // miliseconds
    // std::chrono::duration<double, std::milli> duration2 = end_time2 - start_time2;
    // std::cout << "duration222222: " << duration2.count() << " ms" << std::endl;

#define launch_groupedmm(subgroup_problem_count, iter, subgroup_offset)        \
  {                                                                            \
    int threadblock_count = GemmGrouped::sufficient(                           \
        &problem_sizes[subgroup_offset], subgroup_problem_count);              \
    typename GemmGrouped::Arguments args(                                      \
        problem_sizes_device + subgroup_offset, subgroup_problem_count,        \
        threadblock_count, epilogue_op, in_feat_ptr,                           \
        weight_ptr_array_device + subgroup_offset, C_ptr, out_feat_ptr,        \
        in_feat_indices_device + subgroup_offset, nullptr,                     \
        out_feat_indices_device + subgroup_offset, lda + subgroup_offset,      \
        ldb + subgroup_offset, ldd + subgroup_offset, ldd + subgroup_offset,   \
        input_channels, in_feat_indices_p2[iter],                              \
        problem_sizes + subgroup_offset);                                      \
                                                                               \
    GemmGrouped gemm;                                                          \
    cutlass::Status status;                                                    \
    status = gemm.initialize(args);                                            \
    if (status != cutlass::Status::kSuccess) {                                 \
      std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel."         \
                << std::endl;                                                  \
    }                                                                          \
    status = gemm.run();                                                       \
    if (status != cutlass::Status::kSuccess) {                                 \
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;  \
    }                                                                          \
  }
    // cudaDeviceSynchronize();
    // auto start_time = std::chrono::high_resolution_clock::now();

    // All subgroupedMMs have 3 MMs except the last one: 14-1=13
    for (int subgroup_id = 0; subgroup_id < subgroup_number - 1;
         subgroup_id++) {
      launch_groupedmm(3, subgroup_id, (subgroup_id * 3));
    }
    // last MM have only one MM
    launch_groupedmm(1, (subgroup_number - 1), (subgroup_number - 1) * 3);

    // cudaDeviceSynchronize();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end_time - start_time;
    // std::cout << "duration: " << duration.count() << " ms" << std::endl;
  } else {
    // TODO(tge): not 27
    assert(0);
  }
  return out_feat;
}


at::Tensor conv_forward_fused_cutlass_subgroupedmm_fakeColumn_cuda(at::Tensor in_feat,
                                                                  at::Tensor kernel,
                                                                  const int kernel_volume,
                                                                  const int input_channels,
                                                                  const int output_channels,
                                                                  at::Tensor neighbor_map,
                                                                  at::Tensor neighbor_offset,
                                                                  const int output_size,
                                                                  const bool GatherA,
                                                                  const bool ScatterD,
                                                                  const bool AtomicD,
                                                                  const bool transpose,
                                                                  const int config_no,
                                                                  const bool isHalf,
                                                                  at::Tensor mminfo,
                                                                  at::Tensor mminfo_cpu
                                                                  ) {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int AlignmentA = 8;
  const int AlignmentB = 8;

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  const GroupScheduleMode mode =
      GroupScheduleMode::kDeviceOnly; // kHostPrecompute

  const int kStages = 2;

  const bool fakeColumn = true;

#define GEMM_KERNEL(GatherA, ScatterD, AtomicD, datatype)                      \
  using GemmKernel##GatherA##0##ScatterD##AtomicD =                            \
      typename cutlass::gemm::kernel::DefaultGemmGroupedGatherScatter<         \
          datatype, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,     \
          datatype, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,     \
          datatype, LayoutC, datatype, cutlass::arch::OpClassTensorOp,         \
          cutlass::arch::Sm80, CTAShape, WarpShape, InstructionShape,          \
          cutlass::epilogue::thread::LinearCombination<                        \
              datatype, 128 / cutlass::sizeof_bits<datatype>::value, datatype, \
              datatype>,                                                       \
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,   \
          kStages, GatherA, false, ScatterD, AtomicD, fakeColumn,              \
          mode>::GemmKernel;

#define GEMM_KERNEL_CONFIG(CTAShapeX, CTAShapeY, CTAShapeZ, WarpShapeX,        \
                           WarpShapeY, WarpShapeZ, InstShapeX, InstShapeY,     \
                           InstShapeZ, datatype, torch_datatype)               \
  {                                                                            \
    using CTAShape =                                                           \
        cutlass::gemm::GemmShape<CTAShapeX, CTAShapeY, CTAShapeZ>;             \
    using WarpShape =                                                          \
        cutlass::gemm::GemmShape<WarpShapeX, WarpShapeY, WarpShapeZ>;          \
    using InstructionShape =                                                   \
        cutlass::gemm::GemmShape<InstShapeX, InstShapeY, InstShapeZ>;          \
    GEMM_KERNEL(0, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 1, datatype)                                             \
    if (!GatherA && !ScatterD && !AtomicD) {                                   \
      return launch_cutlass_subgroupedmm_fakeColumn_cuda<                      \
          GemmKernel0000, datatype, torch_datatype>(                           \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, AtomicD,      \
          mminfo, mminfo_cpu);                                                 \
    } else if (GatherA && !ScatterD && !AtomicD) {                             \
      return launch_cutlass_subgroupedmm_fakeColumn_cuda<                      \
          GemmKernel1000, datatype, torch_datatype>(                           \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, AtomicD,      \
          mminfo, mminfo_cpu);                                                 \
    } else if (GatherA && ScatterD && !AtomicD) {                              \
      return launch_cutlass_subgroupedmm_fakeColumn_cuda<                      \
          GemmKernel1010, datatype, torch_datatype>(                           \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, AtomicD,      \
          mminfo, mminfo_cpu);                                                 \
    } else if (GatherA && ScatterD && AtomicD) {                               \
      return launch_cutlass_subgroupedmm_fakeColumn_cuda<                      \
          GemmKernel1011, datatype, torch_datatype>(                           \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, AtomicD,      \
          mminfo, mminfo_cpu);                                                 \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
    assert(false);                                                             \
    return launch_cutlass_subgroupedmm_fakeColumn_cuda<                        \
        GemmKernel0000, datatype, torch_datatype>(                             \
        in_feat, kernel, kernel_volume, input_channels, output_channels,       \
        neighbor_map, neighbor_offset, output_size, transpose, AtomicD,        \
        mminfo, mminfo_cpu);                                                   \
  }

  if (isHalf) {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         cutlass::half_t, at::Half)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         cutlass::half_t, at::Half)
      break;
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  } else {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         float, float)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
      break;
    case 3:
      GEMM_KERNEL_CONFIG(64, 128, 16, // CTAShape
                         32, 64, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
      break;
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  }
}


//////////////////////////////////////////////// cutlass grouped mm
template <typename type_groupmm, typename datatype, typename torch_datatype>
at::Tensor launch_cutlass_groupedmm_cuda( at::Tensor &in_feat,
                                          at::Tensor &kernel,
                                          const int kernel_volume,
                                          const int input_channels,
                                          const int output_channels,
                                          at::Tensor &neighbor_map, // CPU
                                          at::Tensor &neighbor_offset,
                                          const int output_size,
                                          const bool transpose,
                                          at::Tensor mminfo,
                                          at::Tensor mminfo_cpu) {
  // cudaDeviceSynchronize();
  // auto start_time_0 = std::chrono::high_resolution_clock::now();
  using InputType = datatype;
  using OutputType = datatype;
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  at::Tensor out_feat = torch::zeros({output_size, kernel.size(-1)}, options);


  int **in_feat_indices_device;
  int **out_feat_indices_device;
  InputType **weight_ptr_array_device;
  cutlass::gemm::GemmCoord *problem_sizes_device;
  int64_t *lda;
  int64_t *ldb;
  int64_t *ldd;

  int align_test = 32;

  int global_mem_size =
      align_test * (2 * sizeof(int *) + sizeof(InputType *) +
                       sizeof(cutlass::gemm::GemmCoord) + 3 * sizeof(int64_t));
  in_feat_indices_device = reinterpret_cast<int **>(mminfo.data_ptr<uint8_t>());
  out_feat_indices_device =
      in_feat_indices_device + align_test;
  weight_ptr_array_device = reinterpret_cast<InputType **>(out_feat_indices_device + align_test);
  problem_sizes_device = reinterpret_cast<cutlass::gemm::GemmCoord *>(weight_ptr_array_device + align_test);
  lda = reinterpret_cast<int64_t *>(problem_sizes_device + align_test);
  ldb = lda + align_test;
  ldd = ldb + align_test;

  using LayoutA = typename type_groupmm::LayoutA;
  using LayoutB = typename type_groupmm::LayoutB;
  using LayoutC = typename type_groupmm::LayoutC;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<type_groupmm>;
  typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(OutputType(1),
                                                             OutputType(0));

  int *neighbor_offset_ptr = neighbor_offset.data_ptr<int>();


  InputType *in_feat_ptr =
      reinterpret_cast<InputType *>(in_feat.data_ptr<torch_datatype>());
  InputType *weight_ptr =
      reinterpret_cast<InputType *>(kernel.data_ptr<torch_datatype>());
  OutputType *out_feat_ptr =
      reinterpret_cast<OutputType *>(out_feat.data_ptr<torch_datatype>());

  int **in_feat_indices =
      reinterpret_cast<int **>(mminfo_cpu.data_ptr<uint8_t>());
  int **out_feat_indices = in_feat_indices + align_test;
  InputType **weight_ptr_array =
      reinterpret_cast<InputType **>(out_feat_indices + align_test);
  cutlass::gemm::GemmCoord *problem_sizes =
      reinterpret_cast<cutlass::gemm::GemmCoord *>(weight_ptr_array +
                                                   align_test);
  int64_t *lda_host = reinterpret_cast<int64_t *>(problem_sizes + align_test);
  int64_t *ldb_host = lda_host + align_test;
  int64_t *ldd_host = ldb_host + align_test;

  int neighbor_map_size = neighbor_map.size(1);
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose) {
    in_map_ptr = neighbor_map.data_ptr<int>() + neighbor_map_size;
    out_map_ptr = neighbor_map.data_ptr<int>();
  } else {
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + neighbor_map_size;
  }

  int cur_offset = 0;
  int nz_num = 0;
  int neighbor_map_width;
  if (transpose) {
    neighbor_map_width = output_size;
  } else {
    neighbor_map_width = in_feat.size(0);
  }
  // printf("neighbor_map_width: %d\n", neighbor_map_width);
  for (int i = 0; i < kernel_volume; i++) {
    if (neighbor_offset_ptr[i] > 0) {
      int cur = neighbor_offset_ptr[i];
      in_feat_indices[nz_num] = &in_map_ptr[cur_offset];
      out_feat_indices[nz_num] = &out_map_ptr[cur_offset];
      weight_ptr_array[nz_num] = &weight_ptr[i * input_channels * output_channels];
      problem_sizes[nz_num] = cutlass::gemm::GemmCoord(cur, output_channels, input_channels);
      lda_host[nz_num] = LayoutA::packed({cur, input_channels}).stride(0);
      ldb_host[nz_num] = LayoutB::packed({input_channels, output_channels}).stride(0);
      ldd_host[nz_num] = LayoutC::packed({cur, output_channels}).stride(0);
      cur_offset += neighbor_map_width;
      nz_num++;
    }
  }

  cudaMemcpy(in_feat_indices_device, in_feat_indices, global_mem_size,
             cudaMemcpyHostToDevice);
  //   cudaDeviceSynchronize();
  //   auto end_time_0 = std::chrono::high_resolution_clock::now();
  //   printf("init time: %f ms\n",
  //          std::chrono::duration<double, std::milli>(end_time_0 -
  //          start_time_0)
  //              .count());
  
  int problem_count = nz_num;
  int threadblock_count =
      GemmGrouped::sufficient(problem_sizes, problem_count);

  // Configure GEMM arguments
  typename GemmGrouped::Arguments args(
      problem_sizes_device, 
      problem_count, 
      threadblock_count, 
      epilogue_op,
      in_feat_ptr, 
      weight_ptr_array_device, // Weights are not shared
      nullptr, 
      out_feat_ptr, 
      in_feat_indices_device,
      nullptr, 
      out_feat_indices_device, 
      lda, 
      ldb,
      ldd, // nullptr,
      ldd, 
      input_channels,
      nullptr,
      problem_sizes);

  GemmGrouped gemm;
  cutlass::Status status;
  size_t workspace_size = gemm.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
  status = gemm.initialize(args, workspace.get());
  // status = gemm.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel."
              << std::endl;
  }
  // cudaDeviceSynchronize();
  // auto start_time = std::chrono::high_resolution_clock::now();
  status = gemm.run();
  // cudaDeviceSynchronize();
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> duration = end_time - start_time;
  // std::cout << "duration: " << duration.count() << " ms" << std::endl;
  // if (status != cutlass::Status::kSuccess) {
  //   std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
  // }
  return out_feat;
}

at::Tensor conv_forward_fused_cutlass_groupedmm_cuda(at::Tensor in_feat,
                                                     at::Tensor kernel,
                                                     const int kernel_volume,
                                                     const int input_channels,
                                                     const int output_channels,
                                                     at::Tensor neighbor_map,
                                                     at::Tensor neighbor_offset,
                                                     const int output_size,
                                                     const bool GatherA,
                                                     const bool ScatterD,
                                                     const bool AtomicD,
                                                     const bool transpose,
                                                     const int config_no,
                                                     const bool isHalf,
                                                     at::Tensor mminfo,
                                                     at::Tensor mminfo_cpu
                                                     ) {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int AlignmentA = 8;
  const int AlignmentB = 8;

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  const GroupScheduleMode mode =
      GroupScheduleMode::kDeviceOnly; // kHostPrecompute

  const int kStages = 2; // TODO(tge) =2 +gather will faster, specialized mma

  const bool fakeColumn = false;

#define GEMM_KERNEL(GatherA, ScatterD, AtomicD, datatype)                      \
  using GemmKernel##GatherA##0##ScatterD##AtomicD =                            \
      typename cutlass::gemm::kernel::DefaultGemmGroupedGatherScatter<         \
          datatype, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,     \
          datatype, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,     \
          datatype, LayoutC, datatype, cutlass::arch::OpClassTensorOp,         \
          cutlass::arch::Sm80, CTAShape, WarpShape, InstructionShape,          \
          cutlass::epilogue::thread::LinearCombination<                        \
              datatype, 128 / cutlass::sizeof_bits<datatype>::value, datatype, \
              datatype>,                                                       \
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,   \
          kStages, GatherA, false, ScatterD, AtomicD, fakeColumn,              \
          mode>::GemmKernel;

#define GEMM_KERNEL_CONFIG(CTAShapeX, CTAShapeY, CTAShapeZ, WarpShapeX,        \
                           WarpShapeY, WarpShapeZ, InstShapeX, InstShapeY,     \
                           InstShapeZ, datatype, torch_datatype)               \
  {                                                                            \
    using CTAShape =                                                           \
        cutlass::gemm::GemmShape<CTAShapeX, CTAShapeY, CTAShapeZ>;             \
    using WarpShape =                                                          \
        cutlass::gemm::GemmShape<WarpShapeX, WarpShapeY, WarpShapeZ>;          \
    using InstructionShape =                                                   \
        cutlass::gemm::GemmShape<InstShapeX, InstShapeY, InstShapeZ>;          \
    GEMM_KERNEL(0, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 1, datatype)                                             \
    if (!GatherA && !ScatterD && !AtomicD) {                                   \
      return launch_cutlass_groupedmm_cuda<GemmKernel0000, datatype,           \
                                           torch_datatype>(                    \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, mminfo,       \
          mminfo_cpu);                                                         \
    } else if (GatherA && !ScatterD && !AtomicD) {                             \
      return launch_cutlass_groupedmm_cuda<GemmKernel1000, datatype,           \
                                           torch_datatype>(                    \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, mminfo,       \
          mminfo_cpu);                                                         \
    } else if (GatherA && ScatterD && !AtomicD) {                              \
      return launch_cutlass_groupedmm_cuda<GemmKernel1010, datatype,           \
                                           torch_datatype>(                    \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, mminfo,       \
          mminfo_cpu);                                                         \
    } else if (GatherA && ScatterD && AtomicD) {                               \
      return launch_cutlass_groupedmm_cuda<GemmKernel1011, datatype,           \
                                           torch_datatype>(                    \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, transpose, mminfo,       \
          mminfo_cpu);                                                         \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
    assert(false);                                                             \
    return launch_cutlass_groupedmm_cuda<GemmKernel0000, datatype,             \
                                         torch_datatype>(                      \
        in_feat, kernel, kernel_volume, input_channels, output_channels,       \
        neighbor_map, neighbor_offset, output_size, transpose, mminfo,         \
        mminfo_cpu);                                                           \
  }

  if (isHalf) {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         cutlass::half_t, at::Half)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         cutlass::half_t, at::Half)
      break;
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  } else {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         float, float)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
      break;
    case 3:
      GEMM_KERNEL_CONFIG(64, 128, 16, // CTAShape
                         32, 64, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
      break;
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  }
}

///////////////////////////////////////////// cutlass mm
template <typename type_mm, typename datatype, typename torch_datatype>
at::Tensor
launch_cutlass_mm_cuda(at::Tensor &in_feat, at::Tensor &kernel,
                       const int kernel_volume, const int input_channels,
                       const int output_channels, at::Tensor &neighbor_map,
                       at::Tensor &neighbor_offset, const int output_size,
                       std::vector<uint> cuda_streams, const bool transpose) {
  using InputType = datatype;
  using OutputType = datatype;
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  at::Tensor out_feat = torch::zeros({output_size, kernel.size(-1)}, options);

  int in_nnz = in_feat.size(0);
  int out_nnz = out_feat.size(0);
  int in_channel = in_feat.size(1);
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);

  InputType *in_feat_ptr =
      reinterpret_cast<InputType *>(in_feat.data_ptr<torch_datatype>());
  InputType *weight_ptr =
      reinterpret_cast<InputType *>(kernel.data_ptr<torch_datatype>());
  OutputType *out_feat_ptr =
      reinterpret_cast<OutputType *>(out_feat.data_ptr<torch_datatype>());
  OutputType *C_ptr = nullptr;
  if (type_mm::atomicD == true) {
    C_ptr = nullptr;
  } else {
    C_ptr = out_feat_ptr;
  }

  int neighbor_map_size = neighbor_map.size(1);
  // TODO(tge): transpose
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose) {
    in_map_ptr = neighbor_map.data_ptr<int>() + neighbor_map_size;
    out_map_ptr = neighbor_map.data_ptr<int>();
  } else {
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + neighbor_map_size;
  }

  int *neighbor_offset_ptr = neighbor_offset.data_ptr<int>();

  using ElementAccumulator = typename type_mm::ElementAccumulator;
  // Initialize alpha/beta for dot product computation
  ElementAccumulator alpha = ElementAccumulator(1);
  ElementAccumulator beta = ElementAccumulator(1);
  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  int cur_offset = 0;
  int neighbor_map_width;
  if (transpose) {
    neighbor_map_width = output_size;
  } else {
    neighbor_map_width = in_feat.size(0);
  }
  // printf("neighbor_map_width: %d\n", neighbor_map_width);

  for (int k = 0; k < k_vol; k++) {
    int cur_nnz = neighbor_offset_ptr[k];
    if (cur_nnz == 0) {
      continue;
    }
    int cuda_stream_id = k % cuda_streams.size();
    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(cuda_streams[cuda_stream_id]);

    cutlass::gemm::GemmCoord problem_size_real(cur_nnz, out_channel,
                                               in_channel);
    
    // {
    //   // Check map
    //   int *in_map_host = new int[cur_nnz];
    //   int *out_map_host = new int[cur_nnz];
    //   cudaMemcpy(in_map_host, in_map_ptr + cur_offset, cur_nnz * sizeof(int),
    //              cudaMemcpyDeviceToHost);
    //   cudaMemcpy(out_map_host, out_map_ptr + cur_offset, cur_nnz * sizeof(int),
    //              cudaMemcpyDeviceToHost);
    //   for (int i = 0; i < cur_nnz; i++) {
    //     if (in_map_host[i] < 0 || in_map_host[i] >= in_nnz) {
    //       printf("ERROR----- in_map_host[%d]=%d > %d\n", i, in_map_host[i], in_nnz);
    //       break;
    //     }
    //     if (out_map_host[i] < 0 || out_map_host[i] >= out_nnz) {
    //       printf("ERROR----- out_map_host[%d]=%d > %d\n", i, out_map_host[i], out_nnz);
    //       break;
    //     }
    //   }
    //   // for (int i = 0; i < cur_nnz; i++) {
    //   //   printf("in_map_host[%d]=%d, out_map_host[%d]=%d\n", i, in_map_host[i], i, out_map_host[i]);
    //   // }
    //   delete[] in_map_host;
    //   delete[] out_map_host;
    // }

    // Create a tuple of gemm kernel arguments. This is later passed as
    // arguments to launch instantiated CUTLASS kernel

    
    typename type_mm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size_real, // <- problem size of matrix multiplication
        split_k_slices,    // <- k-dimension split factor
        {alpha, beta},     // <- alpha, beta
        in_feat_ptr,       // <- reference to matrix A on device
        &weight_ptr[k * in_channel *
                    out_channel], // <- reference to matrix B on device
        C_ptr,      // <- reference to matrix C on device
        out_feat_ptr, // <- reference to matrix D on device
        cur_nnz * in_channel,     //   options.index_size * problem_size.k(),
        in_channel * out_channel, //   problem_size.k() * problem_size.n(),
        in_nnz * out_channel,     //   problem_size.m() * problem_size.n(),
        in_nnz * out_channel,     //   problem_size.m() * problem_size.n(),
        cutlass::layout::RowMajor::Stride(in_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        cutlass::layout::RowMajor::Stride(out_channel),
        &in_map_ptr[cur_offset],
        nullptr, // <- pointer to index vector to gather  B on device
        &out_map_ptr[cur_offset], // <- pointer to index vector to scatter D on device
    }; 


    type_mm gemm_op;
    cutlass::Status status;
    size_t workspace_size = type_mm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = gemm_op.initialize(arguments, workspace.get());

    // status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS GEMM kernel."
                << std::endl;
    }
    status = gemm_op(cuda_stream);
    // status = gemm_op();
    // cudaDeviceSynchronize();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS GEMM kernel."
                << std::endl;
    }
    cur_offset += neighbor_map_width;
  }
  return out_feat;
}

at::Tensor conv_forward_fused_cutlass_mm_cuda(at::Tensor in_feat,
                                              at::Tensor kernel,
                                              const int kernel_volume,
                                              const int input_channels,
                                              const int output_channels,
                                              at::Tensor neighbor_map,
                                              at::Tensor neighbor_offset,
                                              const int output_size,
                                              const bool GatherA,
                                              const bool ScatterD,
                                              const bool AtomicD,
                                              std::vector<uint> cuda_streams,
                                              const bool transpose,
                                              const int config_no,
                                              const bool isHalf
                                              ) {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int AlignmentA = 8;
  const int AlignmentB = 8;

  const int kStages = 2;

#define GEMM_KERNEL(GatherA, ScatterD, AtomicD, datatype)                      \
  using GemmKernel##GatherA##0##ScatterD##AtomicD =                            \
      cutlass::gemm::device::GemmUniversal<                                    \
          datatype, LayoutA, datatype, LayoutB, datatype, LayoutC, datatype,   \
          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, CTAShape,       \
          WarpShape, InstructionShape,                                         \
          cutlass::epilogue::thread::LinearCombination<                        \
              datatype, 128 / cutlass::sizeof_bits<datatype>::value, datatype, \
              datatype>,                                                       \
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,        \
          kStages, AlignmentA, AlignmentB, cutlass::arch::OpMultiplyAdd,       \
          cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,  \
          GatherA, false, ScatterD, cutlass::layout::NoPermute,                \
          cutlass::layout::NoPermute, cutlass::layout::NoPermute, AtomicD>;

#define GEMM_KERNEL_CONFIG(CTAShapeX, CTAShapeY, CTAShapeZ, WarpShapeX,        \
                           WarpShapeY, WarpShapeZ, InstShapeX, InstShapeY,     \
                           InstShapeZ, datatype, torch_datatype)               \
  {                                                                            \
    using CTAShape =                                                           \
        cutlass::gemm::GemmShape<CTAShapeX, CTAShapeY, CTAShapeZ>;             \
    using WarpShape =                                                          \
        cutlass::gemm::GemmShape<WarpShapeX, WarpShapeY, WarpShapeZ>;          \
    using InstructionShape =                                                   \
        cutlass::gemm::GemmShape<InstShapeX, InstShapeY, InstShapeZ>;          \
    GEMM_KERNEL(0, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 0, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 0, datatype)                                             \
    GEMM_KERNEL(1, 1, 1, datatype)                                             \
    if (!GatherA && !ScatterD && !AtomicD) {                                   \
      return launch_cutlass_mm_cuda<GemmKernel0000, datatype, torch_datatype>( \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, cuda_streams,            \
          transpose);                                                          \
    } else if (GatherA && !ScatterD && !AtomicD) {                             \
      return launch_cutlass_mm_cuda<GemmKernel1000, datatype, torch_datatype>( \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, cuda_streams,            \
          transpose);                                                          \
    } else if (GatherA && ScatterD && !AtomicD) {                              \
      return launch_cutlass_mm_cuda<GemmKernel1010, datatype, torch_datatype>( \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, cuda_streams,            \
          transpose);                                                          \
    } else if (GatherA && ScatterD && AtomicD) {                               \
      return launch_cutlass_mm_cuda<GemmKernel1011, datatype, torch_datatype>( \
          in_feat, kernel, kernel_volume, input_channels, output_channels,     \
          neighbor_map, neighbor_offset, output_size, cuda_streams,            \
          transpose);                                                          \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
    assert(false);                                                             \
    return launch_cutlass_mm_cuda<GemmKernel0000, datatype, torch_datatype>(   \
        in_feat, kernel, kernel_volume, input_channels, output_channels,       \
        neighbor_map, neighbor_offset, output_size, cuda_streams, transpose);  \
  }

  if (isHalf) {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         cutlass::half_t, at::Half)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         cutlass::half_t, at::Half)
      break;
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  } else {
    switch (config_no) {
    case 1:
      GEMM_KERNEL_CONFIG(64, 64, 32, // CTAShape
                         32, 32, 32, // WarpShape
                         16, 8, 8,   // InstShape
                         float, float)
      break;
    case 2:
      GEMM_KERNEL_CONFIG(128, 64, 16, // CTAShape
                         64, 32, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
      break;
    case 3:
      GEMM_KERNEL_CONFIG(64, 128, 16, // CTAShape
                         32, 64, 16,  // WarpShape
                         16, 8, 8,    // InstShape
                         float, float)
    default:
      printf("ERROR: config_no %d not supported\n", config_no);
    }
  }
}
