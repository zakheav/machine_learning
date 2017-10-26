#include <iostream>
#include <vector>
#include "include/Graph.h"
#include "include/node/Input.h"
#include "include/node/Mult.h"
#include "include/node/Add.h"
#include "include/node/Node.h"
#include "include/node/ScalarMult.h"
#include "include/node/Sigmoid.h"
#include "include/node/Trace.h"
using namespace std;
int main() {
    // 构建计算图
    vector<vector<float> > v1(1, vector<float>(1));
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1; ++j) {
            v1[i][j] = 3;
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
    graph.add_node ("", new Input("input1", h, 1));
    graph.add_node ("", new Input("input2", w, 1));
    //Node* mult = new Mult ("mult");
    //graph.add_node ("input1", mult);
    //graph.add_node ("input2", mult);
    //Node* trace = new Trace ("trace");
    //graph.add_node ("mult", trace);    
    Node* scalar_mult = new ScalarMult ("scalar_mult");
    graph.add_node ("input1", scalar_mult);
    graph.add_node ("input2", scalar_mult);

    graph.build_reverse_graph ();// 构建计算图的转置图
    graph.forward_propagation ().display ();// 输出前向传播结果
    graph.back_propagation ();// 进行反向传播
    // 输出对于每个input的导数
    graph.node_map["input1"] -> output -> display ();
    graph.node_map["input2"] -> output -> display ();
}
