#include <iostream>
#include <vector>
#include "include/Graph.h"
#include "include/node/Input.h"
#include "include/node/Mult.h"
#include "include/node/Add.h"
#include "include/node/Node.h"
#include "include/node/ScalarMult.h"
#include "include/node/SquareSum.h"
#include "include/node/Sigmoid.h"
#include "include/node/Trace.h"
using namespace std;
int main() {
    // 构建计算图
    Tensor* x = new Tensor (4, 2);
    x -> tensor[0][0] = 0; x -> tensor[0][1] = 0;
    x -> tensor[1][0] = 0; x -> tensor[1][1] = 1;
    x -> tensor[2][0] = 1; x -> tensor[2][1] = 0;
    x -> tensor[3][0] = 1; x -> tensor[3][1] = 1;
    
    Tensor* w1 = new Tensor (2, 2);
    for (int i = 0; i < w1 -> shape[0]; ++i) {
        for (int j = 0; j < w1 -> shape[1]; ++j) {
            w1 -> tensor[i][j] = (i + j) ; 
        }
    }

    Tensor* w2 = new Tensor (2, 1);
    for (int i = 0; i < w2 -> shape[0]; ++i) {
        for (int j = 0; j < w2 -> shape[1]; ++j) {
            w1 -> tensor[i][j] = (i + j);
        }
    }

    Tensor* one = new Tensor (1, 1);
    one -> tensor[0][0] = -1.0;

    Tensor* y = new Tensor (4, 1);
    y -> tensor[0][0] = 0.0;
    y -> tensor[1][0] = 1.0;
    y -> tensor[2][0] = 1.0;
    y -> tensor[3][0] = 0.0;

    // 构造运算节点
    Node* input_x = new Input ("input_x", x, 0);
    Node* input_w1 = new Input ("input_w1", w1, 1);
    Node* input_w2 = new Input ("input_w2", w2, 1);
    Node* input_one = new Input ("input_one", one, 0);
    Node* input_y = new Input ("input_y", y, 0);
    Node* H1 = new Mult ("H1");
    Node* sigmoid1 = new Sigmoid ("sigmoid1");
    Node* H2 = new Mult ("H2");
    Node* sigmoid2 = new Sigmoid ("sigmoid2");
    Node* minus_y = new ScalarMult ("minus_y");
    Node* h = new Add ("h");
    Node* loss = new SquareSum ("loss");
  
    Graph graph = Graph ();
    // 构造运算图
    graph.add_node ("", input_x);
    graph.add_node ("", input_w1);
    graph.add_node ("input_x", H1);
    graph.add_node ("input_w1", H1);
    graph.add_node ("H1", sigmoid1);
    graph.add_node ("", input_w2);
    graph.add_node ("sigmoid1", H2);
    graph.add_node ("input_w2", H2);
    graph.add_node ("H2", sigmoid2);
    graph.add_node ("", input_one);
    graph.add_node ("", input_y);
    graph.add_node ("input_one", minus_y);
    graph.add_node ("input_y", minus_y);
    graph.add_node ("sigmoid2", h);
    graph.add_node ("minus_y", h);
    graph.add_node ("h", loss);
  
    graph.build_reverse_graph ();// 构建计算图的转置图
    for (int i = 0; i < 100000; ++i) {
        Tensor result = graph.forward_propagation ();// 前向传播
        if (i % 1000 == 0) {
            cout << "error"; result.display (); cout << endl;// 输出前向传播结果
        }
        graph.back_propagation ();// 进行反向传播
    }
    cout << "result: w" << endl;
    input_w1 -> output -> display ();
    input_w2 -> output -> display ();

    // 验证结果
    // 设置计算图中的一些节点隐藏
    input_one -> end_node = 1;
    input_y -> end_node = 1;
    minus_y -> end_node = 1;
    h -> end_node = 1;
    loss -> end_node = 1;
    
    cout << "模型输出：" << endl;
    graph.forward_propagation ().display ();// 验证结果   
}
