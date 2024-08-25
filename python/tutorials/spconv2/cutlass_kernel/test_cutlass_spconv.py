import torch

from cutlass_spconv import conv_forward_fused_cutlass_groupedmm_cuda


if __name__ == "__main__":
    datatype = "fp32"
    M = 5120
    N = 128
    K = 128

    group_size = 4
    num_points = 10000
    assert M <= num_points, f"M({M}) should be less than num_points({num_points})"


    if datatype == "fp32":
        torch_dtype = torch.float32
    elif datatype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError("Invalid datatype")

    input_features = torch.rand((num_points, K), device="cuda", dtype=torch_dtype)
    output_features = torch.empty((num_points, N), device="cuda", dtype=torch_dtype)
    weights = torch.rand((group_size, K, N), device="cuda", dtype=torch_dtype)

    nbmaps = torch.zeros((2, group_size, M), dtype=torch.int32, device="cuda")
    nbsizes_cpu = torch.tensor([M] * group_size, dtype=torch.int32)

    for i in range(group_size):
        A_row_index = torch.randperm(num_points, device="cuda", dtype=torch.int32)[:M]
        C_row_index = torch.randperm(num_points, device="cuda", dtype=torch.int32)[:M]
        nbmaps[0, i, :] = A_row_index
        nbmaps[1, i, :] = C_row_index

    mminfo_mem = torch.ByteTensor(102400).cuda()
    mminfo_cpu = torch.ByteTensor(102400)

    feats_out = conv_forward_fused_cutlass_groupedmm_cuda(
        input_features,
        weights,
        group_size,
        N,
        K,
        nbmaps,
        nbsizes_cpu,
        M,  # output_size
        True,  # GatharA
        True,  # ScatterD
        True,  # AtomicD
        False,  # transposed
        1,  # config_no,
        False,  # isHalf
        mminfo_mem,
        mminfo_cpu,
    )
    print(feats_out)
