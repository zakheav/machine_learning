# -*- coding: UTF-8 -*-
import string
from math import log

dataSet = []  # dataSet中每个元素是一个列表，最后一列是标记


def load_data():
    f = open("samples.txt", "r")
    while True:
        line = f.readline()
        if line:
            str_ = line.strip("\n")
            list_ = str_.split(",")
            data = []
            for e in list_:
                data.append(string.atoi(e))
            dataSet.append(data)
        else:
            break
    f.close()


def get_empirical_entropy(data_set):  # 计算经验熵
    data_num = len(data_set)
    label_dict = {}  # 记录每个类的数目
    for data in data_set:  # 统计每个类的数目
        label = data[-1]  # data最后一个元素，即标签
        if label in label_dict.keys():
            label_dict[label] += 1
        else:
            label_dict[label] = 1
    shannon_entropy = 0.0  # 计算信息熵
    for label in label_dict.keys():
        prob = float(label_dict[label]) / data_num
        shannon_entropy -= log(prob, 2) * prob
    return shannon_entropy


def get_information_gain(data_set, feature_idx):  # 获取信息增益，data_set分割结果
    empirical_entropy = get_empirical_entropy(data_set)  # 经验熵
    sub_dataset_set = {}  # data_set由指定的特征分割出来的子集集合
    for data in data_set:
        if data[feature_idx] in sub_dataset_set.keys():
            sub_dataset_set[data[feature_idx]].append(data)
        else:
            sub_dataset_set[data[feature_idx]] = []
            sub_dataset_set[data[feature_idx]].append(data)
    empirical_conditional_entropy = 0.0  # 经验条件熵
    for feature_val in sub_dataset_set.keys():
        prob = float(len(sub_dataset_set[feature_val])) / len(data_set)
        empirical_conditional_entropy += prob * get_empirical_entropy(sub_dataset_set[feature_val])
    return empirical_entropy - empirical_conditional_entropy, sub_dataset_set


def get_majority_label(data_set):  # 获得占大多数的类别标签
    label_dict = {}  # 记录每个类的数目
    for data in data_set:  # 统计每个类的数目
        label = data[-1]  # data最后一个元素，即标签
        if label in label_dict.keys():
            label_dict[label] += 1
        else:
            label_dict[label] = 1
    max_ = -1
    majority_label = 0
    for label in label_dict.keys():
        if max_ < label_dict[label]:
            max_ = label_dict[label]
            majority_label = label
    return majority_label


def build_node(data_set, node, feature_idx_set):  # 构建每个节点
    if len(feature_idx_set) == 0:  # 属性表为空
        node["leaf"] = True
        node["classification_feature"] = None
        node["child_node"] = None
        node["label"] = get_majority_label(data_set)
    else:
        biggest_information_gain = 0
        best_data_set_divide_result = {}
        best_feature_idx = 0
        for feature_idx in feature_idx_set:  # 遍历属性
            information_gain, data_set_divide_result = get_information_gain(data_set, feature_idx)
            if biggest_information_gain < information_gain:
                biggest_information_gain = information_gain
                best_data_set_divide_result = data_set_divide_result
                best_feature_idx = feature_idx
        if biggest_information_gain < 0.01:
            node["leaf"] = True
            node["classification_feature"] = None
            node["child_node"] = None
            node["label"] = get_majority_label(data_set)
        else:
            feature_idx_set.remove(best_feature_idx)  # 删除已经选择的特征
            node["leaf"] = False
            node["classification_feature"] = best_feature_idx
            node["child_node"] = {}
            node["label"] = None
            for label_val in best_data_set_divide_result:
                node["child_node"][label_val] = {}
                build_node(best_data_set_divide_result[label_val], node["child_node"][label_val], feature_idx_set)


def build_decision_tree(data_set):  # 构建决策树
    feature_idx_set = []  # 特征集合
    feature_num = len(data_set[0]) - 1
    for i in range(feature_num):
        feature_idx_set.append(i)
    tree = {}  # 最终生成的树
    # 树的结构是：
    # {
    #   leaf: yes/no
    #   classification_feature: xxx
    #   child_node: {feature_val1:node, feature_val2:node, ...}
    #   label: 叶子节点存储类型标签
    # }
    build_node(data_set, tree, feature_idx_set)
    return tree


def decision(tree, sample):  # 根据决策树对输入的样本进行决策
    node = tree
    while not node["leaf"]:
        classification_feature = node["classification_feature"]
        node = node["child_node"][sample[classification_feature]]
    return node["label"]


load_data()
the_tree = build_decision_tree(dataSet)
print the_tree
the_sample = [0, 1, 1, 0]
print decision(the_tree, the_sample)
