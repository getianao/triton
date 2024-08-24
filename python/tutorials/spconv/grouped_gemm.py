import triton
import triton.language as tl
from config import get_cuda_autotune_config, _early_config_prune

import cupy as cp
import torch
import functools


def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple, dtype):
    # print(device_ptr, nbytes, shape)
    if dtype == tl.float32:
        cp_dtype = cp.float32
    elif dtype == tl.float16:
        cp_dtype = cp.float16
    else:
        raise ValueError("Invalid datatype")
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=cp_dtype, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")


def _pre_hook(args, **kwargs):
    group_c_ptrs = args[2]  # hard code
    group_gemm_sizes = args[3]  # hard code
    dtype = kwargs["out_dtype"]  # hard code
    for tp_index, tp in enumerate(group_c_ptrs):
        t = ptr_to_tensor(
            tp.item(),
            4 * group_gemm_sizes[tp_index * 3 : tp_index * 3 + 2].prod().item(),
            tuple(group_gemm_sizes[tp_index * 3 : tp_index * 3 + 2].tolist()),
            dtype,
        )
        t.zero_()


@triton.autotune(
    configs=get_cuda_autotune_config(),
    warmup=25, rep=100,
    key=['group_size', 'N_single', 'K_single'],
    pre_hook=_pre_hook,
    prune_configs_by={
        'early_config_prune': functools.partial(_early_config_prune, is_weight=False),
    },
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    M_acc, N_single, K_single,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    in_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(in_dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(in_dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(out_dtype))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b, out_dtype=in_dtype)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(out_dtype)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles
