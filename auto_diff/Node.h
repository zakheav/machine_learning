#ifndef NODE_H_
#define NODE_H_
#include <string>
#include <vector>
/*节点的基类*/
class Node {
    public:
        std::string op_name;// 节点名字（全局唯一）
        std::vector<float> inputs;// 输入
        std::vector<Node*> parents;// 输入节点
        float output;// 输出
        float sum_grad;
        virtual void op ();// 该计算节点的运算函数
        virtual void grad_op ();// 该计算节点对于输入的导函数
        Node ();
        Node (std::string name);
};
#endif
