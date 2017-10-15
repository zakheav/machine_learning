#ifndef GRAPH_H_
#define GRAPH_H_
#include <unordered_map>
#include <vector>
#include <queue>
#include "Node.h"
/*计算图*/
class Graph {
    public:
        std::unordered_map<std::string, Node*> node_map;// 计算图中节点字典
        std::unordered_map<std::string, std::vector<Node*> > adj_table;// 计算图邻接表
        std::unordered_map<std::string, std::vector<Node*> > reverse_table;// 计算图的转置图
        void add_node (std::string parent_name, Node* node);// 向计算图中添加节点
        // 拓扑排序 
        void topological_sort (std::unordered_map<std::string, std::vector<Node*> > &adj_table, std::vector<Node*> &result);
        void build_reverse_graph ();// 构建转置图
        float forward_propagation ();// 前向传播
        void back_propagation ();// 反向传播
        std::string to_string ();
        ~Graph ();
};
#endif
