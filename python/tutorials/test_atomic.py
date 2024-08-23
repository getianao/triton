import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,  
    a_output_ptr,
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Optional meta-parameters for the kernel
):
    # BLOCK_SIZE = meta['BLOCK_SIZE'] 

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # a_val = tl.sum(x, axis = 0)
    # tl.atomic_add(a_output_ptr, a_val)
    tl.atomic_add(x_ptr + offsets, 1)

def add(x: torch.Tensor):
    a_output = torch.zeros(1, device='cuda:0')
    assert x.is_cuda and a_output.is_cuda
    n_elements = x.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, a_output, n_elements, BLOCK_SIZE=128)
    return a_output

size = 256
x = torch.rand(size, device='cuda')
print(x)
output_triton = add(x)
print(x)
print(">>>>")
print("Triton sum: ", print(output_triton))
print("Actual sum: ", x.sum())