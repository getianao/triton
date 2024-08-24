
import triton
from triton.runtime import driver
import torch

def get_cuda_autotune_config():
    tile_sizes_m = [64]
    tile_sizes = [16, 32, 64, 128, 256]
    num_warps_options = [4, 8]  # Possible values for num_warps
    num_stages_options = [3, 4]  # Possible values for num_stages

    configs = []
    triton.Config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
            "NUM_SM": 38,
        },
        num_stages=4,
        num_warps=4,
    ),
    for tile_size_m in tile_sizes_m:
        for tile_size_n in tile_sizes:
            for tile_size_k in tile_sizes:
                for num_warps in num_warps_options:
                    for num_stages in num_stages_options:
                        config = triton.Config(
                            {
                                "BLOCK_SIZE_M": tile_size_m,
                                "BLOCK_SIZE_N": tile_size_n,
                                "BLOCK_SIZE_K": tile_size_k,
                                "NUM_SM": 38,
                            },
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                        configs.append(config)

    return configs

def _early_config_prune(
    configs: triton.Config, named_args: dict, is_weight: bool, **kwargs
):

    pruned_configs = []
    element_size = kwargs["in_dtype"].primitive_bitwidth
    M_acc = named_args["M_acc"]
    N_single = named_args["N_single"]
    K_single = named_args["K_single"]

    device = torch.cuda.current_device()
    min_BLOCK_SIZE_N = min([config.kwargs["BLOCK_SIZE_N"] for config in configs])
    min_BLOCK_SIZE_K = min([config.kwargs["BLOCK_SIZE_K"] for config in configs])
    max_shared_memory = driver.active.utils.get_device_properties(device)[
        "max_shared_mem"
    ]
    for config in configs:
        kw = config.kwargs
        BLOCK_SIZE_M = kw["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = kw["BLOCK_SIZE_N"]
        BLOCK_SIZE_K = kw["BLOCK_SIZE_K"]
        # 1. Prune too large tiles
        if (BLOCK_SIZE_K > K_single and BLOCK_SIZE_K != min_BLOCK_SIZE_K) or (
            BLOCK_SIZE_N > N_single and BLOCK_SIZE_N != min_BLOCK_SIZE_N
        ):
            # print("delete 1")
            continue
        # # 2. Prune configs by shared memory usage
        # required_shared_memory = (
        #     (BLOCK_SIZE_K + BLOCK_SIZE_N)
        #     * BLOCK_SIZE_M
        #     * config.num_stages
        #     * element_size
        # )
        # # if required_shared_memory > max_shared_memory:
        # #     # print("delete 2", required_shared_memory, max_shared_memory)
        # #     continue
        # # 3. Prune configs with large tile sizes and small warp sizes (register pressure)
        # if BLOCK_SIZE_K >= 256 and BLOCK_SIZE_N >= 256 and config.num_warps == 4:
        #     # print("delete 3")
        #     continue
        pruned_configs.append(config)
    if len(pruned_configs) == 0:
        pruned_configs.append(configs[0])
    if True:
        print(
            f"Number of configs pruned from {len(configs)} to {len(pruned_configs)}, is_weight={is_weight}"
        )
        # print(pruned_configs)
    return pruned_configs