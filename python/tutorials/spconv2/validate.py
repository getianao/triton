import torch
import triton
import triton.language as tl

from benchmark import cutlass_perf_fn, triton_atomic_perf_fn

if __name__ == "__main__":
    datatype="fp32"

    if datatype == "fp32":
        torch_dtype = torch.float32
        tl_dtype = tl.float32
    elif datatype == "fp16":
        torch_dtype = torch.float16
        tl_dtype = tl.float16
    else:
        raise ValueError("Invalid datatype")

    M = 5120
    N = 128
    K = 128

    group_size = 4
    num_points = 5120
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

    nbmaps = torch.zeros((2, group_size, num_points), dtype=torch.int32, device="cuda")
    nbsizes_cpu = torch.tensor([M] * group_size, dtype=torch.int32)
    for i in range(group_size):
        nbmaps[0, i, :M] = A_row_index_group[i]
        nbmaps[1, i, :M] = C_row_index_group[i]
    nbmaps = nbmaps.view(2, group_size * num_points).contiguous()
    mminfo_mem = torch.ByteTensor(102400).cuda()
    mminfo_cpu = torch.ByteTensor(102400)

    cutlass_output = cutlass_perf_fn(
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
    )

    torch.cuda.synchronize()

    triton_atomic_perf_fn(
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
    )

    print("Cutlass output: ", cutlass_output)
    print("Triton output: ", output_features)

    if torch.allclose(cutlass_output, output_features, atol=1e-2, rtol=1e-2) == False:
        print("Error at", i)
        print("Expected: ", cutlass_output)
        print("Got: ", output_features)
        # assert False and "Validation failed"
    else:
        print("Validation succeeded")
