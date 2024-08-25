"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

import triton
import triton.language as tl

from kernel_grouped_gemm_atomic import grouped_matmul_kernel_atomic

from cutlass_spconv import conv_forward_fused_cutlass_groupedmm_cuda


def triton_atomic_perf_fn(
    input_features,
    weights,
    output_features,
    sizes,
    a_row_index_ptrs,
    c_row_index_ptrs,
    lds,
    group_size,
    M_acc,
    N_single,
    K_single,
    tl_dtype: tl.constexpr,
    atomic_c: tl.constexpr,
):
    grid = lambda META: (META["NUM_SM"],)
    grouped_matmul_kernel_atomic[grid](
        input_features,
        weights,
        output_features,
        sizes,
        a_row_index_ptrs,
        c_row_index_ptrs,
        lds,
        group_size,
        M_acc, N_single, K_single,
        in_dtype = tl_dtype,
        out_dtype = tl_dtype,
        atomic_c = atomic_c,
    )

def cutlass_perf_fn(
    input_features,
    weights,
    output_features,
    sizes,
    group_size,
    M_acc,
    N_single,
    K_single,
    num_points,
    nbmaps,
    nbsizes_cpu,
    mminfo_mem,
    mminfo_cpu,
    tl_dtype: tl.constexpr,
    atomic_c: tl.constexpr,
):
    feats_out = conv_forward_fused_cutlass_groupedmm_cuda(
        input_features,
        weights,
        group_size,
        N_single,
        K_single,
        nbmaps,
        nbsizes_cpu,
        num_points,  # output_size
        True,  # GatharA
        True,  # ScatterD
        atomic_c,  # AtomicD
        False,  # transposed
        1,  # config_no,
        False,  # isHalf
        mminfo_mem,
        mminfo_cpu,
    )
    return feats_out

def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)

if __name__ == '__main__':
    datatype = "fp32"
    print("Data type: ", datatype)
    # validate(datatype = datatype)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=["M", "N", "K"],
            x_vals=[
                (5120, 2**i, 2**i) for i in range(4, 10)
            ],  # different possible values for `x_name`
            line_arg="provider",
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=["triton", "triton_atomic", "cutlass"],
            # label name for the lines
            line_names=["Triton", "Triton_Atomic", "Cutlass"],
            # line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="tflops",  # label name for the y-axis
            plot_name="group-gemm-performance",
            # name for the plot. Used also as a file name for saving the plot.
            args={"datatype": datatype},  # default parameters
        )
    )
    def benchmark(M, N, K, provider, datatype):
        group_size = 4
        num_points = 10000
        assert M <= num_points, f"M({M}) should be less than num_points({num_points})"

        g_sizes = []
        g_lds = []
        A_row_index_addrs = []
        C_row_index_addrs = []
        A_row_index_group = [] # to keep indices tensors alive
        C_row_index_group = [] # to keep indices tensors alive

        if datatype == "fp32":
            torch_dtype = torch.float32
            tl_dtype = tl.float32
        elif datatype == "fp16":
            torch_dtype = torch.float16
            tl_dtype = tl.float16
        else:
            raise ValueError("Invalid datatype")

        input_features = torch.rand((num_points, K), device="cuda", dtype=torch_dtype)
        output_features = torch.empty((num_points, N), device="cuda", dtype=torch_dtype)
        weights = torch.rand((group_size, K, N), device="cuda", dtype=torch_dtype)

        for i in range(group_size):
            A_row_index = torch.randperm(num_points, device="cuda", dtype=torch.int32)[:M]
            C_row_index = torch.randperm(num_points, device="cuda", dtype=torch.int32)[:M]
            A_row_index_addrs.append(A_row_index.data_ptr())
            C_row_index_addrs.append(C_row_index.data_ptr())
            A_row_index_group.append(A_row_index) 
            C_row_index_group.append(C_row_index)
            g_sizes += [M, N, K]
        g_lds = [input_features.stride(0), weights.stride(1), output_features.stride(0)]
        d_a_row_index_ptrs = torch.tensor(A_row_index_addrs, device="cuda")
        d_c_row_index_ptrs = torch.tensor(C_row_index_addrs, device="cuda")
        d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
        d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

        M_acc = sum([g_sizes[3 * i] for i in range(group_size)])
        N_single = g_sizes[1]
        K_single = g_sizes[2]
        # print("M_acc", M_acc, "N_single", N_single, "K_single", K_single)

        quantiles = [0.5, 0.2, 0.8]
        # if provider == "cublas":
        #     ms, min_ms, max_ms = triton.testing.do_bench(
        #         lambda: torch_perf_fn(group_A, group_B),
        #         quantiles=quantiles,
        #         warmup=25,
        #         rep=100,
        #     )
        if provider == "cutlass":
            nbmaps = torch.zeros((2, group_size, num_points), dtype=torch.int32, device="cuda")
            nbsizes_cpu = torch.tensor([M] * group_size, dtype=torch.int32)
            for i in range(group_size):
                nbmaps[0, i, :M] = A_row_index_group[i]
                nbmaps[1, i, :M] = C_row_index_group[i]
            nbmaps = nbmaps.view(2, group_size * num_points).contiguous()
            mminfo_mem = torch.ByteTensor(102400).cuda()
            mminfo_cpu = torch.ByteTensor(102400)

            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: cutlass_perf_fn(
                    input_features,
                    weights,
                    output_features,
                    d_g_sizes,
                    group_size,
                    M_acc,
                    N_single,
                    K_single,
                    num_points,
                    nbmaps,
                    nbsizes_cpu,
                    mminfo_mem,
                    mminfo_cpu,
                    tl_dtype=tl_dtype,
                    atomic_c=True,
                ),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_atomic_perf_fn(
                    input_features,
                    weights,
                    output_features,
                    d_g_sizes,
                    d_a_row_index_ptrs,
                    d_c_row_index_ptrs,
                    d_g_lds,
                    group_size,
                    M_acc,
                    N_single,
                    K_single,
                    tl_dtype=tl_dtype,
                    atomic_c=False,
                ),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )
        if provider == "triton_atomic":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_atomic_perf_fn(
                    input_features,
                    weights,
                    output_features,
                    d_g_sizes,
                    d_a_row_index_ptrs,
                    d_c_row_index_ptrs,
                    d_g_lds,
                    group_size,
                    M_acc,
                    N_single,
                    K_single,
                    tl_dtype=tl_dtype,
                    atomic_c=True,
                ),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )

        def perf(time):
            flop = 0
            for i in range(group_size):
                flop += 2 * g_sizes[3 * i] * g_sizes[3 * i + 1] * g_sizes[3 * i + 2]
            return flop / time * 1e-9

        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
