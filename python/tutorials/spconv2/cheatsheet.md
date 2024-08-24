MLIR_ENABLE_DUMP=grouped_matmul_kernel_atomic  python benchmark.py 2>&1 | tee dump.mlir

TRITON_PRINT_AUTOTUNING=1 python benchmark.py