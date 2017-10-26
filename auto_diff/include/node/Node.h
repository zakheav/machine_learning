#ifndef NODE_H_
#define NODE_H_
#include <string>
#include <vector>
#include "../Tensor.h"
/*节点的基类*/
class Node {
    public:
        std::string op_name;// 节点名字（全局唯一）
        std::vector<Node*> parents;// 输入节点
        Tensor* output;// 输出
        Tensor* sum_grad;
        int need_update;
        int end_node;
        float a;// 学习率
        virtual void op ();// 该计算节点的运算函数
        virtual void grad_op ();// 该计算节点对于输入的导函数
        virtual void update ();// 根据梯度更新
        Node ();
        Node (std::string name);
        ~Node ();
};
#endif
