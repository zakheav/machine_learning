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
    Tensor* w = new Tensor (v1);
    cout << "w:" << endl;
    w -> display ();

    vector<vector<float> > v2(1, vector<float>(1));
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1; ++j) {
            v2[i][j] = 2;
        }
    }
    Tensor* x = new Tensor (v2);
    cout << "x:" << endl;
    x -> display ();

    vector<vector<float> > v3(1, vector<float>(1));
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1; ++j) {
            v3[i][j] = 10;
        }
    }
    Tensor* y = new Tensor (v3);
    cout << "label:" << endl;
    y -> display();

    vector<vector<float> > v4(1, vector<float>(1));
    v4[0][0] = -1.0;
    Tensor* one = new Tensor (v4);

    Node* input_w = new Input ("input_w", w, 1);
    Node* input_x = new Input ("input_x", x, 0);
    Node* wx = new Mult ("wx");
    Node* input_y = new Input ("input_y", y, 0);
    Node* input_one = new Input ("input_one", one, 0);
    Node* minus_y = new ScalarMult ("minus_y");
    Node* h1 = new Add ("h1");
    Node* h2 = new Add ("h2");
    Node* loss = new Mult ("loss");

    Graph graph = Graph ();
    graph.add_node ("", input_w);
    graph.add_node ("", input_x);
    graph.add_node ("", input_y);
    graph.add_node ("", input_one);
    graph.add_node ("input_w", wx);
    graph.add_node ("input_x", wx);
    graph.add_node ("input_one", minus_y);
    graph.add_node ("input_y", minus_y);
    graph.add_node ("wx", h1);
    graph.add_node ("minus_y", h1);
    graph.add_node ("wx", h2);
    graph.add_node ("minus_y", h2);
    graph.add_node ("h1", loss);
    graph.add_node ("h2", loss);

    graph.build_reverse_graph ();// 构建计算图的转置图
    for (int i = 0; i < 5; ++i) {
        cout << "error: ";
        graph.forward_propagation ().display ();// 输出前向传播结果
        cout << endl;
        graph.back_propagation ();// 进行反向传播
    }
    cout << "result: w" << endl;
    input_w -> output -> display ();
}
