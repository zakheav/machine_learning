#include "Graph.h"
#include <iostream>
#include <vector>
#include "Input.h"
#include "Mult.h"
#include "Add.h"
#include "Node.h"
#include "Sigmoid.h"
#include "Trace.h"
using namespace std;
int main() {
    // 构建计算图
    vector<vector<float> > v1(2, vector<float>(2));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            v1[i][j] = i + j * i;
        }
    }
    Tensor* h = new Tensor (v1);
    h -> display ();

    vector<vector<float> > v2(2, vector<float>(2));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            v2[i][j] = i + j + 10;
        }
    }
    Tensor* w = new Tensor (v2);
    w -> display ();

    Graph graph = Graph ();
    graph.add_node ("", new Input("input1", h));
    graph.add_node ("", new Input("input2", w));
    //Node* mult = new Mult ("mult");
    //graph.add_node ("input1", mult);
    //graph.add_node ("input2", mult);
    //Node* trace = new Trace ("trace");
    //graph.add_node ("mult", trace);    
    Node* add = new Add ("add");
    graph.add_node ("input1", add);
    graph.add_node ("input2", add);

    graph.build_reverse_graph ();// 构建计算图的转置图
    graph.forward_propagation ().display ();// 输出前向传播结果
    graph.back_propagation ();// 进行反向传播
    // 输出对于每个input的导数
    graph.node_map["input1"] -> sum_grad -> display ();
    graph.node_map["input2"] -> sum_grad -> display ();
}
