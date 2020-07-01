# coding=utf-8
import numpy as np
import cv2
import time
import math
import numba
from numba import cuda


# 用于对影像进行扩边，返回扩边后的影像，更方便卷积处理
# 核心是调用了numpy的矩阵拼接函数实现扩边，边界灰度为0
def preprocessing(img, filter):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    img_h = img.shape[0]
    img_w = img.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2

    margin_w = np.zeros([img_h, half_filter_w])
    margin_h = np.zeros([half_filter_h, img_w + 2 * half_filter_w])
    tmp_img = np.hstack((margin_w, img, margin_w))
    extend_img = np.vstack((margin_h, tmp_img, margin_h))
    return extend_img


# 最原始的卷积算法，遍历所有像素
def filterImg(extend_img, filter):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2
    img_h = extend_img.shape[0]
    img_w = extend_img.shape[1]

    filtered_img = np.zeros_like(img)

    # 在遍历时尤其需要注意像素索引的问题，因为这个很容易错，而且一错就会导致各种问题
    for i in range(half_filter_w + 1, img_w - half_filter_w):
        for j in range(half_filter_h + 1, img_h - half_filter_h):
            patch = extend_img[j - half_filter_h:j + half_filter_h + 1, i - half_filter_w:i + half_filter_w + 1]
            filter_value = int(np.sum(patch * filter))
            filtered_img[j - half_filter_h, i - half_filter_w] = filter_value
    return filtered_img


# 利用Numba CPU加速的卷积
# 和上面的代码是一样的，只是函数前加了Numba的前缀
@numba.jit(nopython=True)
def filterImgNumba(extend_img, filter):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2
    img_h = extend_img.shape[0]
    img_w = extend_img.shape[1]

    filtered_img = np.zeros_like(img)

    for i in range(half_filter_w + 1, img_w - half_filter_w):
        for j in range(half_filter_h + 1, img_h - half_filter_h):
            patch = extend_img[j - half_filter_h:j + half_filter_h + 1, i - half_filter_w:i + half_filter_w + 1]
            filter_value = int(np.sum(patch * filter))
            filtered_img[j - half_filter_h, i - half_filter_w] = filter_value
    return filtered_img


# 利用CUDA进行GPU加速
# 经过初步测试CUDA对于读取一些越界的索引，返回0，不会报错，但是如果对越界的索引赋值可能会引发报错
@cuda.jit
def filterImgGPU(input_img, filter, filtered_img):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2

    x, y = cuda.grid(2)  # 获取当前线程在整个grid中的位置
    # 在安全索引范围内遍历
    if x < filtered_img.shape[0] and y < filtered_img.shape[1]:
        if half_filter_h < x < filtered_img.shape[0] - half_filter_h and \
                half_filter_w < y < filtered_img.shape[1] - half_filter_w:
            # 获取patch
            patch = input_img[x - half_filter_h:x + half_filter_h + 1, y - half_filter_w:y + half_filter_w + 1]
            tmp = 0.0
            # 各元素相乘累加
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    tmp += patch[j, i] * filter[j, i]
            filtered_img[x, y] = tmp
        else:
            # 目前的策略是，对于边界不进行处理，原值返回
            filtered_img[x, y] = input_img[x, y]


# 相比于上一个函数，这个函数接受的是经过扩边的影像
# 这个函数中尤其要注意的就是扩边后的影像与原始影像的坐标之间的换算关系
@cuda.jit
def filterImgGPU2(extend_img, filter, filtered_img):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2

    # 主要流程和上面是一样的
    x, y = cuda.grid(2)
    if half_filter_h < x < extend_img.shape[0] - half_filter_h and \
            half_filter_w < y < extend_img.shape[1] - half_filter_w:
        patch = extend_img[x - half_filter_h:x + half_filter_h + 1, y - half_filter_w:y + half_filter_w + 1]
        tmp = 0.0
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                tmp += patch[j, i] * filter[j, i]
        # 这个地方就涉及到了扩边后的影像与原始影像的坐标之间的换算关系，注意理解
        filtered_img[x - half_filter_h, y - half_filter_w] = tmp


if __name__ == '__main__':
    # 卷积相关设置与影像读取
    kernel_size = 21  # 卷积核大小
    # 构造一个21×21的卷积核用于影像模糊
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    img = cv2.imread("test.jpg")  # 读取影像
    img_b, img_g, img_r = cv2.split(img)  # 拆分RGB通道
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为对应灰度图
    # 对影像进行扩边，方便后续卷积操作
    img_b_ext = preprocessing(img_b, kernel)
    img_g_ext = preprocessing(img_g, kernel)
    img_r_ext = preprocessing(img_r, kernel)
    img_gray_ext = preprocessing(img_gray, kernel)

    # 每个Block中的线程个数(一维)
    TPB = 16

    # 将卷积核复制到device上
    kernel_device = cuda.to_device(kernel)

    # 原图的device变量以及结果
    img_b_device = cuda.to_device(img_b)
    img_g_device = cuda.to_device(img_g)
    img_r_device = cuda.to_device(img_r)
    result_b_device = cuda.device_array((img.shape[0], img.shape[1]))
    result_g_device = cuda.device_array((img.shape[0], img.shape[1]))
    result_r_device = cuda.device_array((img.shape[0], img.shape[1]))

    # 扩边后的device变量以及结果
    img_b_device_ext = cuda.to_device(img_b_ext)
    img_g_device_ext = cuda.to_device(img_g_ext)
    img_r_device_ext = cuda.to_device(img_r_ext)
    result_b_device_ext = cuda.device_array((img.shape[0], img.shape[1]))
    result_g_device_ext = cuda.device_array((img.shape[0], img.shape[1]))
    result_r_device_ext = cuda.device_array((img.shape[0], img.shape[1]))

    # 方法1，最原始的卷积遍历
    # --------------------------------------------------------------------------------
    start_time = time.time()
    filtered_cpu = filterImg(img_gray_ext, kernel)
    end_time = time.time()
    print 'cpu time', end_time - start_time
    # --------------------------------------------------------------------------------

    # 方法2，Numba CPU加速的卷积遍历
    # --------------------------------------------------------------------------------
    start_time = time.time()
    filtered_numba = filterImgNumba(img_gray_ext, kernel)
    end_time = time.time()
    print 'cpu time(Numba)', end_time - start_time
    # --------------------------------------------------------------------------------

    # 方法3，CUDA GPU加速的卷积遍历
    # --------------------------------------------------------------------------------
    threadsperblock = (TPB, TPB)
    # 注意这里两个shape都是int，所以得到的结果还是int，且有可能出现0
    # 这样的结果就是构造的线程数小于图像大小，某些地方会出现空白
    # 解决办法是将运算变成float类型，结果就是小数，再向上取整即可
    blockspergrid_x = int(math.ceil(1.0 * img_b.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(1.0 * img_b.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    filterImgGPU[blockspergrid, threadsperblock](img_b_device, kernel_device, result_b_device)
    filterImgGPU[blockspergrid, threadsperblock](img_g_device, kernel_device, result_g_device)
    filterImgGPU[blockspergrid, threadsperblock](img_r_device, kernel_device, result_r_device)
    end_time = time.time()

    print 'gpu time*3', end_time - start_time

    result_b_host = result_b_device.copy_to_host()
    result_g_host = result_g_device.copy_to_host()
    result_r_host = result_r_device.copy_to_host()
    filtered_gpu1 = cv2.merge((result_b_host, result_g_host, result_r_host))
    # --------------------------------------------------------------------------------

    # 方法4，扩边以后的CUDA GPU加速的卷积遍历
    # --------------------------------------------------------------------------------
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(1.0 * img_b_ext.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(1.0 * img_b_ext.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    filterImgGPU2[blockspergrid, threadsperblock](img_b_device_ext, kernel_device, result_b_device_ext)
    filterImgGPU2[blockspergrid, threadsperblock](img_g_device_ext, kernel_device, result_g_device_ext)
    filterImgGPU2[blockspergrid, threadsperblock](img_r_device_ext, kernel_device, result_r_device_ext)
    end_time = time.time()

    print 'gpu time(extend)*3', end_time - start_time

    result_b_host_ext = result_b_device_ext.copy_to_host()
    result_g_host_ext = result_g_device_ext.copy_to_host()
    result_r_host_ext = result_r_device_ext.copy_to_host()
    filtered_gpu2 = cv2.merge((result_b_host_ext, result_g_host_ext, result_r_host_ext))
    # --------------------------------------------------------------------------------

    # 方法5，OpenCV对照测试
    # --------------------------------------------------------------------------------
    start_time = time.time()
    filtered_cv = cv2.filter2D(img, -1, kernel)
    end_time = time.time()
    print 'opencv time', end_time - start_time
    # --------------------------------------------------------------------------------

    # 保存所有的输出结果
    cv2.imwrite("filtered_cpu.jpg", filtered_cpu)
    cv2.imwrite("filtered_numba.jpg", filtered_numba)
    cv2.imwrite("filtered_gpu.jpg", filtered_gpu1)
    cv2.imwrite("filtered_gpu_ext.jpg", filtered_gpu2)
    cv2.imwrite("filtered_cv.jpg", filtered_cv)
