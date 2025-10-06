import numpy as np
from numba import cuda

@cuda.jit
def square_array(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = arr[idx] * arr[idx]

# Host array
n = 10
arr = np.arange(n, dtype=np.float32)

# Copy to device
d_arr = cuda.to_device(arr)

# Launch kernel
threads_per_block = 32
blocks_per_grid = (arr.size + (threads_per_block - 1)) // threads_per_block
square_array[blocks_per_grid, threads_per_block](d_arr)

# Copy back to host
result = d_arr.copy_to_host()
print(result)