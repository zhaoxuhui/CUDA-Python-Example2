# coding=utf-8
from numba import cuda, float32
import numba
import numpy
import math
import time

TPB = 16


@numba.jit(nopython=True)
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[x, k] * B[k, y]
            C[x, y] = tmp


@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp


A = numpy.full((TPB * 100, TPB * 100), 3, numpy.float)
B = numpy.full((TPB * 100, TPB * 100), 4, numpy.float)
C_cpu = numpy.full((A.shape[0], B.shape[1]), 0, numpy.float)

print("Start processing in CPU")
start_cpu = time.time()
# matmul_cpu(A, B, C_cpu)
end_cpu = time.time()
time_cpu = (end_cpu - start_cpu)
print("CPU time： " + str(time_cpu))

# Start in GPU
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]))

threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print("Start processing in GPU")
start_gpu = time.time()
matmul_gpu[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize()
end_gpu = time.time()
time_gpu = (end_gpu - start_gpu)
print("GPU time(global memory): " + str(time_gpu))
C_global_gpu = C_global_mem.copy_to_host()

sart_gpu_shared = time.time()
matmul_shared_mem[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_shared_mem)
cuda.synchronize()
end_gpu_shared = time.time()

time_gpu_shared = (end_gpu_shared - sart_gpu_shared)
print("GPU time(shared memory): " + str(time_gpu_shared))
C_shared_gpu = C_shared_mem.copy_to_host()
