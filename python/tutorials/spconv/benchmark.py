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

from grouped_gemm import grouped_matmul_kernel
from grouped_gemm_atomic import grouped_matmul_kernel_atomic


def group_gemm_fn(group_A, group_B, tl_dtype: tl.constexpr):
    device = torch.device("cuda")
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META["NUM_SM"],)
    M_acc = sum([g_sizes[3 * i] for i in range(group_size)])
    N_single = g_sizes[1]
    K_single = g_sizes[2]
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        M_acc, N_single, K_single,
        in_dtype = tl_dtype,
        out_dtype = tl_dtype,
    )

    return group_C


def validate(datatype="fp32"):
    if datatype == "fp32":
        torch_dtype = torch.float32
        tl_dtype = tl.float32
    elif datatype == "fp16":
        torch_dtype = torch.float16
        tl_dtype = tl.float16
    else:
        raise ValueError("Invalid datatype")

    group_m = [1024, 2048, 4096]
    group_n = [1024, 2048, 4096]
    group_k = [1024, 2048, 4096]
    group_A = []
    group_B = []
    assert len(group_m) == len(group_n)
    assert len(group_n) == len(group_k)
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device="cuda", dtype=torch_dtype)
        B = torch.rand((K, N), device="cuda", dtype=torch_dtype)
        group_A.append(A)
        group_B.append(B)

    tri_out = group_gemm_fn(group_A, group_B, tl_dtype=tl_dtype)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(group_size):
        if torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2) == False:
            print("Error at", i)
            print("Expected: ", ref_out[i])
            print("Got: ", tri_out[i])
            # assert False and "Validation failed"
        else:
            print("Validation succeeded")


# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    sizes,
    lds,
    group_size,
    M_acc,
    N_single,
    K_single,
    tl_dtype: tl.constexpr,
):
    grid = lambda META: (META["NUM_SM"],)
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
        M_acc,
        N_single,
        K_single,
        in_dtype = tl_dtype,
        out_dtype = tl_dtype,
    )

def triton_atomic_perf_fn(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    sizes,
    lds,
    group_size,
    M_acc,
    N_single,
    K_single,
    tl_dtype: tl.constexpr,
):
    grid = lambda META: (META["NUM_SM"],)
    grouped_matmul_kernel_atomic[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
        M_acc, N_single, K_single,
        in_dtype = tl_dtype,
        out_dtype = tl_dtype,
    )

def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)

if __name__ == '__main__':
    datatype = "fp32"
    print("Data type: ", datatype)
    validate(datatype = datatype)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=["M", "N", "K"],
            # x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
            x_vals=[(10240, 2**i, 2**i) for i in range(4, 10)],
            # x_vals=[(2**i, 2**i, 2**i) for i in range(7, 13)],
            line_arg="provider",
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=["cublas", "triton", "triton_atomic"],
            # label name for the lines
            line_names=["cuBLAS", "Triton", "Triton_Atomic"],
            # line_vals=["triton"],  line_names=["Triton"],
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
        group_A = []
        group_B = []
        A_addrs = []
        B_addrs = []
        C_addrs = []
        g_sizes = []
        g_lds = []
        group_C = []

        if datatype == "fp32":
            torch_dtype = torch.float32
            tl_dtype = tl.float32
        elif datatype == "fp16":
            torch_dtype = torch.float16
            tl_dtype = tl.float16
        else:
            raise ValueError("Invalid datatype")
        for i in range(group_size):
            A = torch.rand((M, K), device="cuda", dtype=torch_dtype)
            B = torch.rand((K, N), device="cuda", dtype=torch_dtype)
            C = torch.empty((M, N), device="cuda", dtype=torch_dtype)  # torch_dtype
            group_A.append(A)
            group_B.append(B)
            group_C.append(C)
            A_addrs.append(A.data_ptr())
            B_addrs.append(B.data_ptr())
            C_addrs.append(C.data_ptr())
            g_sizes += [M, N, K]
            g_lds += [A.stride(0), B.stride(0), C.stride(0)]

        d_a_ptrs = torch.tensor(A_addrs, device="cuda")
        d_b_ptrs = torch.tensor(B_addrs, device="cuda")
        d_c_ptrs = torch.tensor(C_addrs, device="cuda")
        d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
        d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

        M_acc = sum([g_sizes[3 * i] for i in range(group_size)])
        N_single = g_sizes[1]
        K_single = g_sizes[2]
        # print("M_acc", M_acc, "N_single", N_single, "K_single", K_single)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "cublas":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_perf_fn(group_A, group_B),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_perf_fn(
                    d_a_ptrs,
                    d_b_ptrs,
                    d_c_ptrs,
                    d_g_sizes,
                    d_g_lds,
                    group_size,
                    M_acc,
                    N_single,
                    K_single,
                    tl_dtype=tl_dtype,
                ),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )
        if provider == "triton_atomic":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_atomic_perf_fn(
                    d_a_ptrs,
                    d_b_ptrs,
                    d_c_ptrs,
                    d_g_sizes,
                    d_g_lds,
                    group_size,
                    M_acc,
                    N_single,
                    K_single,
                    tl_dtype=tl_dtype,
                ),
                quantiles=quantiles,
                warmup=25,
                rep=100,
            )

        def perf(time):
            flop = 0
            for i in range(group_size):
                # flop += 2 * g_sizes[i]
                flop += (
                    2 * group_A[i].shape[0] * group_A[i].shape[1] * group_B[i].shape[1]
                )
            return flop / time * 1e-9

        return perf(ms), perf(max_ms), perf(min_ms)
        # return (ms), (max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
