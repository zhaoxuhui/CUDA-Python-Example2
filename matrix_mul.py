# coding=utf-8
import numpy as np
from numba import cuda, float32
import numba
import time
import math


# 最原始的矩阵相乘算法
def matmul_cpu(A, B, C):
    # 根据矩阵乘法原理，可以先根据A、B的尺寸算出结果矩阵C的大小
    # 矩阵乘法结果C的大小是A的行数(shape[0])、B的列数(shape[1])
    # 然后循环遍历C中的每个元素，分别获得A、B中对应行列的元素，相乘并累加即可
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp = 0.0
            # 根据矩阵乘法法则，A的列数(shape[1])应该和B的行数(shape[0])相等
            # 所以这里A.shape[1]和B.shape[0]是等价的
            for k in range(A.shape[1]):
                # A、B中行列的对应元素相乘
                tmp += A[i, k] * B[k, j]
            # 最后将累加的结果放到对应位置中，完成一个元素的计算
            C[i, j] = tmp


# 调用Numba开启CPU并行加速
# 其实和原始算法比，代码内容没有做任何改动，只是在函数名称前加了Numba的修饰符即可实现CPU加速
@numba.jit(nopython=True)
def matmul_cpu_numba(A, B, C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


# 通过Numba调用CUDA开启GPU加速
# 在编写CUDA核函数的时候，应该时刻有“并行”的思想
# 每一个线程都会同时执行核函数，因此可以节省一些for循环
@cuda.jit
def matmul_gpu_global_mem(A, B, C):
    # 利用CUDA Python的API获得当前线程在整个grid中的x、y索引
    x, y = cuda.grid(2)
    # 如果索引在有效范围内，没越界的话执行下面代码
    # 实际经过测试发现，就算不写也不会报错，但从代码的完备性上来说还是写一下
    if x < C.shape[0] and y < C.shape[1]:
        tmp = 0.0
        # 和前面类似，得到当前线程位置后，获取A、B中对应行列的元素相乘并累加
        # A.shape[1]和B.shape[0]是等价的
        for i in range(A.shape[1]):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# 利用shared memory加速读写操作，提高速度
# 同一个block中的线程可以共享一小段shared memory，特点是读写非常快，但是内存较小
# 我们可以将矩阵拆分成多个block，每个小块各自先进行乘法，最后再将所有结果累加，这样可以提升速度
@cuda.jit
def matmul_gpu_shared_mem(A, B, C):
    TPB = 16
    # 新建两个小的矩阵块对应block中的线程(所以大小和block一样)，数据类型为CDUA的float32
    tA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    tB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # 先获取到当前线程在整个grid中的坐标
    x, y = cuda.grid(2)

    # 再获取到当前线程在当前block中的坐标
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # 简单进行越界判断
    # 根据矩阵乘法，C的大小是A的行数(shape[0])、B的列数(shape[1])
    if x > A.shape[0] or y > B.shape[1]:
        return

    # 计算一下按照当前tA、tB的大小需要计算多少次
    exec_num = int(math.ceil(1.0 * A.shape[1] / TPB))
    tmp = 0.0
    # 每个线程都执行下面的循环
    for i in range(exec_num):
        # 首先根据当前线程位置，获取到A、B中的对应元素并赋给tA、tB对应位置
        tA[tx, ty] = A[x, ty + i * TPB]
        tB[tx, ty] = B[x + i * TPB, y]
        # 因为核函数是所有线程并行执行的，所以这里我们需要等待一下
        # 等待所有线程都执行完上面的操作后再继续进行
        cuda.syncthreads()

        # 到这里以后，tA、tB中所有的元素都被赋值了
        # 所以我们就按照常规矩阵乘法，每个线程都执行行列对应相乘
        for j in range(TPB):
            tmp += tA[tx, j] * tB[j, ty]
        # 和上面一样，要等待所有线程都执行完以后再继续
        cuda.syncthreads()
        # 执行完一次循环以后，一个拆分的小格子就被计算完了，再移动到下个block计算
        # 上面所有计算的结果都累加到了tmp中

    # 最后将tmp赋给C中的对应位置，结束
    C[x, y] = tmp


if __name__ == '__main__':
    # 待计算的矩阵
    # 需要注意的还是shape的顺序问题，shape[0]表示行方向，shape[1]表示列方向
    A = np.full([700, 500], 3, np.float32)
    B = np.full([500, 700], 5, np.float32)
    # 将矩阵拷贝到GPU中方便处理
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    # 每个Block中的线程个数(一维)
    TPB = 16

    # 需要注意的是，CUDA加速只有在计算数据很大时才会有明显的效果
    # 如果数据量很小，可能比CPU还慢(因为和CPU相比多了大量线程调度等操作)

    # 方法1，最原始的算法
    # --------------------------------------------------------------------------------
    C_cpu_result = np.zeros([A.shape[0], B.shape[1]], np.float32)
    start_time = time.time()
    matmul_cpu(A, B, C_cpu_result)
    end_time = time.time()
    print 'cpu time', end_time - start_time
    # --------------------------------------------------------------------------------

    # 方法2，利用Numba进行CPU加速
    # --------------------------------------------------------------------------------
    C_cpu_result_numba = np.zeros([A.shape[0], B.shape[1]], np.float32)
    start_time = time.time()
    matmul_cpu_numba(A, B, C_cpu_result_numba)
    end_time = time.time()
    print 'cpu time(Numba)', end_time - start_time
    # --------------------------------------------------------------------------------

    # 方法3，利用CUDA进行GPU加速
    # --------------------------------------------------------------------------------
    # 新建一个device上的矩阵用于存放结果
    C_gpu_result_global = cuda.device_array((A.shape[0], B.shape[1]))

    # 进行thread-block-grid配置
    threadsperblock = (TPB, TPB)
    # 注意这里两个shape都是int，所以得到的结果还是int，且有可能出现0
    # 解决办法是将运算变成float类型，结果就是小数，再向上取整即可
    blockspergrid_x = int(math.ceil(1.0 * A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(1.0 * B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    # 手动指定设置的thread-block-grid
    matmul_gpu_global_mem[blockspergrid, threadsperblock](A_device, B_device, C_gpu_result_global)
    # 可以同步等待一下，所有任务做完以后才会继续执行
    cuda.synchronize()
    end_time = time.time()
    # 最后别忘了把device上的结果拷贝回来
    C_cpu_result_global = C_gpu_result_global.copy_to_host()
    print 'gpu time(global memory)', end_time - start_time
    # --------------------------------------------------------------------------------

    # 方法4，利用CUDA和Shared Memory进一步提速
    # 这里需要注意的是在数据量很小时，shared memory的优势可能体现不出来
    # 因为根据设计的算法，是一个个小的block依次计算然后累加
    # 和一个线程对应一个元素相比可能还会变慢
    # --------------------------------------------------------------------------------
    C_gpu_result_shared = cuda.device_array((A.shape[0], B.shape[1]))

    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(1.0 * A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(1.0 * B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    matmul_gpu_shared_mem[blockspergrid, threadsperblock](A_device, B_device, C_gpu_result_shared)
    cuda.synchronize()
    end_time = time.time()
    C_cpu_result_shared = C_gpu_result_shared.copy_to_host()
    print 'gpu time(shared memory)', end_time - start_time
    # --------------------------------------------------------------------------------
