# -*- coding: UTF-8 -*-
# 神经网络的bp算法实现
# 神经网络有m层，第n层的神经元数目为N(n)
# 第n层的输入为第n-1层的输出+偏置项（1），是一个N(n-1)+1维的行向量Xn
# 第n层的权值矩阵为Wn，每个元素Wnij（j>=1）表示第n层的i号单元和第n-i层j号单元的关联权值，Wni0是n层i单元的偏置
# Wn是 N(n)*(N(n-1)+1)维的矩阵
# 第n层的输出Yn =f(Wn * Xn.T)是个N(n)维的行向量，Yni是n层第i个神经元的输出
# 在前向传播过程中，第n层会缓存Yn对Wn的导数矩阵DYWn，i行j列元素为dYni/dWnij，DYWn是 N(n)*(N(n-1)+1)维的矩阵
# 在前像传播过程中，第n层会缓存Yn对Y(n-1)的导数矩阵DYXn，i行j列元素为dYni/dY(n-1)j，DYXn是 N(n)*N(n-1)维的矩阵

# 假如最终的损失函数为F(Ym)，F(Ym)对第m层的各个神经元的输出的偏导向量为(dF/dYm)，假设m=3
# 则，F关于第3层的权值的偏导矩阵DFW3 = [dF/dY3] * DYW3
# F关于第2层的权值的偏导矩阵DFW2 = [dF/dY3 * DYX3] * DYW2
# F关于第1层的权值的偏导矩阵DFW1 = [dF/dY3 * DYX3 * DYX2] * DYW1
# [X_vector]符号表示以X_vector中元素作为对角线的对角矩阵

import string
from numpy import *
import csv

sampleSet = []  # 样本集
labelSet = []  # 样本的标签集
learning_rate = 0.1  # 学习速率


def load_csv(path):
    csvfile = file(path, 'rb')
    reader = csv.reader(csvfile)
    for line in reader:
        label = string.atoi(line[0])
        labelSet.append(label)
        line[0] = 1  # 增广输入向量，以匹配偏置项
        for i in range(len(line)):
            if i >= 1:
                if line[i] == "0":
                    line[i] = 0
                else:
                    line[i] = 1
        sampleSet.append(line)


class ActivationFunction:  # 激活函数对象
    def __init__(self):
        pass

    @staticmethod
    def f(x):  # 激活函数：反曲正切函数
        if x > 20:
            return 1
        if x < -20:
            return -1
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def df(x):  # 激活函数的导数
        if x > 20 or x < -20:
            return 0
        return 1 - math.pow((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), 2)


class LossFunction:  # 最终的损失函数
    def __init__(self):
        pass

    @staticmethod
    def f(x_matrix, label_matrix):  # 损失函数，输入的都是行向量
        loss = 0.0
        for i in range(x_matrix.shape[0]):
            loss += math.pow(x_matrix[0, i] - label_matrix[0, i], 2)
        return loss

    @staticmethod
    def df(x_matrix, label_matrix):  # 损失函数的梯度向量，输入的都是行向量
        grad_matrix = zeros([1, x_matrix.shape[1]])
        for i in range(x_matrix.shape[1]):
            grad_matrix[0, i] = 2 * (x_matrix[0, i] - label_matrix[0, i])
        return matrix(grad_matrix)


class Layer:
    def __init__(self, layer_id, unit_num, input_num):
        self.layer_id = layer_id  # 层号
        self.__unit_num = unit_num  # 该层神经元个数
        self.__input_num = input_num  # 输入的数量（不包含偏置）
        self.W_matrix = mat(random.randn(unit_num, input_num + 1))  # 初始化权值矩阵
        self.dW_matrix = mat(zeros([unit_num, input_num + 1]))  # 在不同的样本下权值修改的累计值
        self.dyw_matrix = mat(zeros([unit_num, input_num + 1]))  # 输出对权值的导数矩阵
        self.dyx_matrix = mat(zeros([unit_num, input_num]))  # 输出对输入的导数矩阵

    def __cache_dywmatrix(self, temp_matrix, input_matrix):  # 在前向传播过程中缓存dyw_matrix
        for i in range(self.dyw_matrix.shape[0]):
            d = ActivationFunction.df(temp_matrix[i, 0])
            for j in range(self.dyw_matrix.shape[1]):
                self.dyw_matrix[i, j] = d * input_matrix[0, j]

    def __cache_dyxmatrix(self, temp_matrix):  # 在前向传播过程中缓存dyx_matrix
        for i in range(self.dyx_matrix.shape[0]):
            d = ActivationFunction.df(temp_matrix[i, 0])
            for j in range(self.dyx_matrix.shape[1]):
                self.dyx_matrix[i, j] = d * self.W_matrix[i, j + 1]

    def forward_propagation(self, input_matrix):  # 前向传播，input_matrix是增广后的输入（行向量），第一位是1
        output_matrix = self.W_matrix * input_matrix.T
        self.__cache_dywmatrix(output_matrix, input_matrix)
        self.__cache_dyxmatrix(output_matrix)
        for i in range(output_matrix.shape[0]):
            for j in range(output_matrix.shape[1]):
                output_matrix[i, j] = ActivationFunction.f(output_matrix[i, j])
        return output_matrix.T

    def backward_propagation(self, backward_input_matrix):  # 反向传播，调整权值矩阵W_matrix，输入为行向量
        backward_output_matrix = backward_input_matrix * self.dyx_matrix
        backward_input_array = []
        for i in range(backward_input_matrix.shape[1]):
            backward_input_array.append(backward_input_matrix[0, i])
        self.dW_matrix += (matrix(diag(backward_input_array)) * self.dyw_matrix)
        return backward_output_matrix


class Network:
    def __init__(self):
        self.network = []
        self.network.append(Layer(1, 10, 4))
        self.network.append(Layer(1, 5, 10))
        self.network.append(Layer(2, 1, 5))
        self.train_time = 1000
        self.threshold = 0.01  # 错误阈值
        self.alpha = 0.1  # 正则项因子

    def train(self):
        counter = 0
        not_finish = True
        error = 0.0
        while counter < self.train_time and not_finish:
            counter += 1
            error = 0.0
            for i in range(len(sampleSet)):  # 遍历所有的样本
                sample = sampleSet[i]
                input_matrix = matrix(array(sample))
                output = None
                for layer in self.network:  # 前向传播
                    output = layer.forward_propagation(input_matrix)
                    input_matrix = c_[output, array([1])]
                error += LossFunction.f(output, matrix(array([labelSet[i]])))
                backward_input_matrix = LossFunction.df(output, matrix(array([labelSet[i]])))

                for layer in self.network[::-1]:  # 反向传播，计算每一层在不同样本下权值修改的累计值dW_matrix
                    backward_input_matrix = layer.backward_propagation(backward_input_matrix)

                for layer in self.network:  # 修改权值矩阵
                    layer.W_matrix -= (layer.dW_matrix * (1.0 / len(sampleSet)) + layer.W_matrix * self.alpha) * learning_rate

                error /= len(sampleSet)
            print error
            if error < self.threshold:
                not_finish = False
        print error

    def run(self, sample):
        sample.insert(0, 1)
        input_matrix = matrix(array(sample))
        output = None
        for layer in self.network:  # 前向传播
            output = layer.forward_propagation(input_matrix)
            input_matrix = c_[output, array([1])]
        print output


load_csv("train.csv")
network = Network()
network.train()

