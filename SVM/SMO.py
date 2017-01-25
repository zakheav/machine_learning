# -*- coding: UTF-8 -*-
# 算法流程：
# 选择两个违反KKT对偶互补的a: a1，a2
# 固定其他的a，最优化函数，得到新的a1new，a2new
# 更新b让a1new，a2new满足KKT条件
# 重复前面的过程直到收敛
# 以下代码中，_m是矩阵的后缀，_a是数组的后缀

import string
from numpy import *

sampleSet = []  # 样本集
labelSet = []  # 样本的标签集
C = 10  # 超参数，用于平衡结构风险和经验风险，C越大，结构风险的比例越重


def load_file():  # 读取文件
    f = open("samples.txt", "r")
    while True:
        line = f.readline()
        if line:
            the_string = line.strip('\n')
            list_ = the_string.split(" ")
            data = []
            for i in range(len(list_)):
                if i < len(list_) - 1:
                    data.append(string.atof(list_[i]))
                else:
                    labelSet.append(string.atoi(list_[i]))
            sampleSet.append(data)
        else:
            break
    f.close()


def kernel_function(n):
    return n


def build_kernel_matrix():  # 构建核矩阵，矩阵元素为kernel_function(<xi, xj>)
    sample_matrix1_m = matrix(sampleSet)
    sample_matrix2_m = matrix(sampleSet).T  # 转置
    kernel_matrix_m = sample_matrix1_m * sample_matrix2_m
    # 用kernel函数处理the_matrix中的每个值
    for i in range(kernel_matrix_m.shape[0]):
        for j in range(kernel_matrix_m.shape[1]):
            kernel_matrix_m[i, j] = kernel_function(kernel_matrix_m[i, j])
    return kernel_matrix_m


def calculate_e(a_vector_m, i, kernel_matrix_m, b):
    # ei = g(xi) - yi
    # g(xi) = ({aj * yj * Kij}对j求和) + b - yi, kij表示kernel_function(<xi,xj>)
    # a_vector_m是行向量
    kernel_array_a = (kernel_matrix_m.getA())[i:]  # 获取kernel_matrix中的第i行向量
    label_set_a = array(labelSet)
    return (a_vector_m * matrix(kernel_array_a * label_set_a).T)[0, 0] + b - labelSet[i]


def choose_a2(a_vector_m, e_vector_m, kernel_matrix_m, b, error):  # 同时会计算e向量
    a2_idx = -1
    for i in range(a_vector_m.shape[1]):
        # 当ai=0时 ei*yi>=0，
        # 当ai=C时 ei*yi<=0，
        # 当C>ai>0时 ei*yi=0
        ai = a_vector_m[0, i]
        e_vector_m[0, i] = calculate_e(a_vector_m, i, kernel_matrix_m, b)
        if C > ai > 0:  # C>ai>0
            if abs(e_vector_m[0, i] * labelSet[i]) > error:
                a2_idx = i
                break
        elif abs(ai - 0) < 0.00001:  # ai=0
            if e_vector_m[0, i] * labelSet[i] < 0:
                a2_idx = i
                break
        elif abs(ai - C) < 0.00001:  # ai=C
            if e_vector_m[0, i] * labelSet[i] > 0:
                a2_idx = i
                break
    return a2_idx


def choose_a1(a2_idx, a_vector_m):  # 在选择出了a2的基础上选择a1
    i = a2_idx
    size_ = a_vector_m.shape[1]
    while i == a2_idx:
        i = int(random.random() * size_)
    return i


def cut_a2(a2, l, h):  # 对a2进行剪裁，满足约束条件
    if a2 > h:
        return h
    if a2 < l:
        return l
    return a2


def calculate_eta(kernel_matrix_m, a1_idx, a2_idx):
    # 这个函数的返回值一定是非负数，否则就是核函数选择有问题（不满足Mercer定理）
    return kernel_matrix_m[a1_idx, a1_idx] + kernel_matrix_m[a2_idx, a2_idx] - 2 * kernel_matrix_m[a1_idx, a2_idx]


def smo(error):  # 输入精度需求，一般是0.0001
    load_file()  # 获取样本
    sample_num = len(sampleSet)
    # 初始化alphaVector，EVector, b
    a_vector_m = matrix(zeros(sample_num))
    e_vector_m = matrix(zeros(sample_num))  # 用于缓存
    b = 0
    # 计算kernelMatrix
    kernel_matrix_m = build_kernel_matrix()

    loop_time = 0
    find_a_violate_kkt = True

    while loop_time < 1000 and find_a_violate_kkt:  # 主循环
        find_a_violate_kkt = False
        # 选择a2，同时计算对应的e2存储在e_vector_m中
        a2_idx = choose_a2(a_vector_m, e_vector_m, kernel_matrix_m, b, error)

        if a2_idx != -1:
            find_a_violate_kkt = True
            a2 = a_vector_m[0, a2_idx]

            a1_idx = choose_a1(a2_idx, a_vector_m)  # 选择a1
            a1 = a_vector_m[0, a1_idx]

            e_vector_m[0, a1_idx] = calculate_e(a_vector_m, a1_idx, kernel_matrix_m, b)  # 计算并缓存e1
            eta = calculate_eta(kernel_matrix_m, a1_idx, a2_idx)  # 计算eta

            if eta <= 0:  # 说明核函数有问题
                print "eta <= 0, kernel function has problem"
                continue

            # 计算L，H
            if labelSet[a1_idx] == labelSet[a2_idx]:
                l = max(0, a1 + a2 - C)
                h = min(C, a1 + a2)
            else:
                l = max(0, a2 - a1)
                h = min(C, a2 - a1 + C)
            if l == h:
                print "L=H", l
                continue

            # 计算a2_new
            a2_new_unc = a2 + labelSet[a2_idx] * (e_vector_m[0, a1_idx] - e_vector_m[0, a2_idx]) / eta
            a2_new = cut_a2(a2_new_unc, l, h)
            # 计算a1_new
            a1_new = a1 + labelSet[a1_idx] * labelSet[a2_idx] * (a2 - a2_new)

            # 更新b，使得a1_new、a2_new满足kkt法则
            b1 = -e_vector_m[0, a1_idx] - labelSet[a1_idx] * kernel_matrix_m[a1_idx, a1_idx] * (a1_new - a1) \
                 - labelSet[a2_idx] * kernel_matrix_m[a2_idx, a1_idx] * (a2_new - a2) + b
            b2 = -e_vector_m[0, a2_idx] - labelSet[a1_idx] * kernel_matrix_m[a1_idx, a2_idx] * (a1_new - a1) \
                 - labelSet[a2_idx] * kernel_matrix_m[a2_idx, a2_idx] * (a2_new - a2) + b
            if C > a1_new > 0:
                b = b1
            elif C > a2_new > 0:
                b = b2
            else:
                b = (b1 + b2) / 2

            a_vector_m[0, a1_idx] = a1_new
            a_vector_m[0, a2_idx] = a2_new

            loop_time += 1

    print "loop time: ", loop_time

    # 计算w向量
    a_array_a = a_vector_m.getA()
    label_set_a = array(labelSet)
    w_vector_m = matrix(a_array_a * label_set_a) * matrix(sampleSet)
    print w_vector_m, b


smo(0.0001)
