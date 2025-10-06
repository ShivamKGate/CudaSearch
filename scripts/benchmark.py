import time
import numpy as np

# example cpu computation
def cpu_sum_of_squares(arr):
    return np.sum(arr ** 2)

# example gpu computation
try:
    import cupy as cp 
    def gpu_sum_of_squares(arr):
        arr_gpu = cp.array(arr)
        result = cp.sum(arr_gpu ** 2)
        return cp.asnumpy(result)
    gpu_available = True
except ImportError:
    gpu_available = False

# random data
size = 10_000_000
data = np.random.rand(size).astype(np.float32)

# Benchmark CPU
start = time.time()
cpu_result = cpu_sum_of_squares(data)
cpu_time = time.time() - start
print(f"CPU result: {cpu_result:.2f}, Time: {cpu_time:.4f} seconds")

# Benchmark GPU
if gpu_available:
    start = time.time()
    gpu_result = gpu_sum_of_squares(data)
    gpu_time = time.time() - start
    print(f"GPU result: {gpu_result:.2f}, Time: {gpu_time:.4f} seconds")
else:
    print("cupy not installed, GPU benchmark skipped.")
