import torch
import ctypes
from math import prod
import cupy as cp

# It additionally needs the ctypes type as torch type
def as_tensor(pointer, shape, torch_type):
    arr = (pointer._type_ * prod(shape)).from_address(
        ctypes.addressof(pointer.contents))
    
    return torch.frombuffer(arr, dtype=torch_type).view(*shape)

def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple):
    print(device_ptr, nbytes, shape)
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=cp.float32, memptr=memptr)
    print(arr)
    return torch.as_tensor(arr, device="cuda")

shape = (1024, 1024, 1024)
x = torch.zeros(shape, dtype=torch.float32, device='cuda')

# p = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))

# print(p)  # Print pointer
# y = as_tensor(p, shape, torch.float)

y = ptr_to_tensor(x.data_ptr(), x.numel() * x.element_size(), shape)

print(y)  # Print created tensor

x[1,1,0] = 3.  # Modify original

print(y)  # Print again