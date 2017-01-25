# -*- coding: UTF-8 -*-
# LR是一个分类模型，假设 y|x,w ～ (phi)^y * (1-phi)^(1-y)  即伯努力分布，y = 0,1
# 模型想要得到 E(y|x,w)，即在已知输入属性的情况下分类结果的期望；如果期望偏向于1，则输出1，否则输出0
# E(y|x,w) = phi，为了让phi满足值域在[0,1]，并且单调递增，令phi = 1/(1+exp(-x*w))
# 把phi代入伯努力分布，得到p(y|x,w) = (phi)^y * (1-phi)^(1-y)
# 使用极大似然估计求解w

# 以下代码中，_m是矩阵的后缀，_a是数组的后缀
from numpy import *
import string

sampleSet = []  # sampleSet中每个元素是一个列表
labelSet = []  # 表示每个样本对应的标签

def load_data():
    f = open("samples.txt", "r")
    while True:
        line = f.readline()
        if line:
            the_string = line.strip("\n")
            list_ = the_string.split(" ")
            data = []
            for i in range(len(list_)):
                if i < len(list_) - 1:
                    data.append(string.atoi(list_[i]))
                else:
                    labelSet.append(string.atoi(list_[i]))
            sampleSet.append(data)
        else:
            break
    f.close()


def get_grad_vector(w_m):  # 获取单位梯度向量
    y_m = matrix(labelSet)
    x_m = matrix(sampleSet)
    w_x_m = w_m * x_m.T
    w_x_a = w_x_m.getA()
    exp_w_x_1 = []
    for i in range(alen(w_x_a)):
        exp_w_x_1.append(1 / (exp(-w_x_a[i]) + 1))
    exp_w_x_1_m = matrix(exp_w_x_1)  # exp(w*x)-1矩阵
    grad_m = exp_w_x_1_m * x_m - y_m * x_m
    return grad_m


def train_lr():
    load_data()
    a = 0.1
    time = 200
    w_m = matrix(zeros(len(sampleSet[0]), float))
    while time > 0:
        grad_m = get_grad_vector(w_m)
        mod_ = sqrt(grad_m * grad_m.T)
        print mod_
        if mod_ < 0.1:
            break
        else:
            unit_grad_m = grad_m.getA() / mod_
            w_m -= unit_grad_m.getA() * a
        time -= 1
        print w_m
    return w_m


def judge(sample, w_m):
    sample_m = matrix(sample)
    h = 1 / (1 + exp(-(sample_m * w_m.T).getA()[0][0]))
    print h

judge([2, 1, 0, 1], train_lr())
