# -*- coding: UTF-8 -*-
# 神经网络的bp算法实现
# 神经网络有m层，第n层的神经元数目为N(n)
# 第n层的输入为第n-1层的输出+偏置项（1），是一个N(n-1)+1维的行向量Xn
# 第n层的权值矩阵为Wn，每个元素Wnij（j>=1）表示第n层的i号单元和第n-i层j号单元的关联权值，Wni0是n层i单元的偏置
# Wn是 N(n)*(N(n-1)+1)维的矩阵
# 第n层的输出Yn = Wn * Xn.T 是个N(n)维的行向量，Yni是n层第i个神经元的输出
# 在前向传播过程中，第n层会缓存Yn对Wn的导数矩阵DYWn，i行j列元素为dYni/dWnij，DYWn是 N(n)*(N(n-1)+1)维的矩阵
# 在前像传播过程中，第n层会缓存Yn对Y(n-1)的导数矩阵DYXn，i行j列元素为dYni/dY(n-1)j，DYXn是 N(n)*N(n-1)维的矩阵

# 假如最终的损失函数为F(Ym)，F(Ym)对第m层的各个神经元的输出的偏导向量为(dF/dYm)，假设m=3
# 则，F关于第3层的权值的偏导矩阵DFW2 = [dF/dY3] * DYW2
# F关于第2层的权值的偏导矩阵DFW2 = [dF/dY3 * DYX3] * DYW2
# F关于第1层的权值的偏导矩阵DFW2 = [dF/dY3 * DYX3 * DYX2] * DYW1
# [X_vector]符号表示以X_vector中元素作为对角线的对角矩阵

import string
from numpy import *

sampleSet = []  # 样本集
labelSet = []  # 样本的标签集
learning_rate = 0.1  # 学习速率


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
                    data.append(1)
                    labelSet.append(string.atoi(list_[i]))
            sampleSet.append(data)
        else:
            break
    f.close()


class ActivationFunction:  # 激活函数对象
    def __init__(self):
        pass

    @staticmethod
    def f(x):  # 激活函数：反曲正切函数
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def df(x):  # 激活函数的导数
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
        self.W_matrix = mat(random.rand(unit_num, input_num + 1))  # 初始化权值矩阵
        self.dyw_matrix = zeros([unit_num, input_num + 1])  # 输出对权值的导数矩阵
        self.dyx_matrix = zeros([unit_num, input_num])  # 输出对输入的导数矩阵

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
        dw_matrix = matrix(diag(backward_input_array)) * self.dyw_matrix * learning_rate
        self.W_matrix -= dw_matrix
        return backward_output_matrix


class Network:
    def __init__(self):
        self.network = []
        self.network.append(Layer(1, 3, 2))
        self.network.append(Layer(2, 3, 3))
        self.network.append(Layer(3, 1, 3))
        self.train_time = 10000
        self.threshold = 0.01

    def train(self):
        counter = 0
        not_finish = True
        while counter < self.train_time and not_finish:
            counter += 1
            error = 0.0
            for i in range(len(sampleSet)):
                sample = sampleSet[i]
                input_matrix = matrix(array(sample))
                output = None
                for layer in self.network:  # 前向传播
                    output = layer.forward_propagation(input_matrix)
                    input_matrix = c_[output, array([1])]
                # print output
                error += LossFunction.f(output, matrix(array([labelSet[i]])))
                backward_input_matrix = LossFunction.df(output, matrix(array([labelSet[i]])))

                for layer in self.network[::-1]:  # 反向传播
                    backward_input_matrix = layer.backward_propagation(backward_input_matrix)
            if error < self.threshold:
                not_finish = False

    def run(self, sample):
        sample.append(1)
        input_matrix = matrix(array(sample))
        output = None
        for layer in self.network:  # 前向传播
            output = layer.forward_propagation(input_matrix)
            input_matrix = c_[output, array([1])]
        print output

load_file()
network = Network()
network.train()
network.run([1, 1])
network.run([1, 0])
network.run([0, 1])
network.run([0, 0])
