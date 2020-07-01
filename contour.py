# coding=utf-8
import numpy as np
import cv2
from numba import cuda
import math
import time


@cuda.jit
def filterImgGPU(input_img, filter, filtered_img):
    filter_h = filter.shape[0]
    filter_w = filter.shape[1]
    half_filter_h = filter_h / 2
    half_filter_w = filter_w / 2

    x, y = cuda.grid(2)
    if x < filtered_img.shape[0] and y < filtered_img.shape[1]:
        if half_filter_h < x < filtered_img.shape[0] - half_filter_h and \
                half_filter_w < y < filtered_img.shape[1] - half_filter_w:
            patch = input_img[x - half_filter_h:x + half_filter_h + 1, y - half_filter_w:y + half_filter_w + 1]
            tmp = 0.0
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    tmp += patch[j, i] * filter[j, i]
            filtered_img[x, y] = tmp
        else:
            filtered_img[x, y] = input_img[x, y]


if __name__ == '__main__':
    # 读取影像并拷贝到device
    img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    img_device = cuda.to_device(img)

    # 构造4个方向的卷积核并拷贝到device
    kernel1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel3 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kernel4 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel1_device = cuda.to_device(kernel1)
    kernel2_device = cuda.to_device(kernel2)
    kernel3_device = cuda.to_device(kernel3)
    kernel4_device = cuda.to_device(kernel4)

    # 新建4个device变量用于存放结果
    result1_device = cuda.device_array((img.shape[0], img.shape[1]))
    result2_device = cuda.device_array((img.shape[0], img.shape[1]))
    result3_device = cuda.device_array((img.shape[0], img.shape[1]))
    result4_device = cuda.device_array((img.shape[0], img.shape[1]))

    # 构造运算架构
    TPB = 16
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(1.0 * img.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(1.0 * img.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()

    # 执行卷积
    filterImgGPU[blockspergrid, threadsperblock](img_device, kernel1_device, result1_device)
    filterImgGPU[blockspergrid, threadsperblock](img_device, kernel2_device, result2_device)
    filterImgGPU[blockspergrid, threadsperblock](img_device, kernel3_device, result3_device)
    filterImgGPU[blockspergrid, threadsperblock](img_device, kernel4_device, result4_device)

    end_time = time.time()

    print 'cost time', end_time - start_time

    # 再将运算结果拷贝回Host
    result_1_host = result1_device.copy_to_host()
    result_2_host = result2_device.copy_to_host()
    result_3_host = result3_device.copy_to_host()
    result_4_host = result4_device.copy_to_host()

    # 直接得到的结果是float类型，而且也不是0到255，所以先将数据转换到合适范围，再转成uint8
    result1_res = np.where(result_1_host < 0, 0, result_1_host)
    result1 = np.uint8(np.where(result1_res > 255, 255, result1_res))
    result2_res = np.where(result_2_host < 0, 0, result_2_host)
    result2 = np.uint8(np.where(result2_res > 255, 255, result2_res))
    result3_res = np.where(result_3_host < 0, 0, result_3_host)
    result3 = np.uint8(np.where(result3_res > 255, 255, result3_res))
    result4_res = np.where(result_4_host < 0, 0, result_4_host)
    result4 = np.uint8(np.where(result4_res > 255, 255, result4_res))

    # 保存各方向的轮廓
    cv2.imwrite("contour1.jpg", result1)
    cv2.imwrite("contour2.jpg", result2)
    cv2.imwrite("contour3.jpg", result3)
    cv2.imwrite("contour4.jpg", result4)

    # 将各方向轮廓合并
    tmp1 = cv2.bitwise_or(result1, result2)
    tmp2 = cv2.bitwise_or(tmp1, result3)
    tmp3 = cv2.bitwise_or(tmp2, result4)
    cv2.imwrite("contour.jpg", tmp3)
