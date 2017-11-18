#include <iostream>
#include <vector>
#include "include/Calculator.h"
using namespace std;
int main() {
    // 构建计算图
    Tensor* x = new Tensor (4, 2);
    x -> tensor[0][0] = 0; x -> tensor[0][1] = 0;
    x -> tensor[1][0] = 0; x -> tensor[1][1] = 1;
    x -> tensor[2][0] = 1; x -> tensor[2][1] = 0;
    x -> tensor[3][0] = 1; x -> tensor[3][1] = 1;
    
    Tensor* w1 = new Tensor (2, 4);

    Tensor* w2 = new Tensor (4, 1);

    Tensor* one = new Tensor (1, 1);
    one -> tensor[0][0] = -1.0;

    Tensor* y = new Tensor (4, 1);
    y -> tensor[0][0] = 0.0;
    y -> tensor[1][0] = 1.0;
    y -> tensor[2][0] = 1.0;
    y -> tensor[3][0] = 0.0;
    
    Calculator cal = Calculator ();
    
    // 构造运算节点
    Node* input_x = cal.input_data (x);
    Node* input_one = cal.input_data (one);
    Node* input_y = cal.input_data (y);
    Node* input_w1 = cal.input_variable (w1);
    Node* input_w2 = cal.input_variable (w2);
    Node* H1 = cal.mult (input_x, input_w1);
    Node* sigmoid1 = cal.sigmoid (H1);
    Node* H2 = cal.mult (sigmoid1, input_w2);
    Node* sigmoid2 = cal.sigmoid (H2);
    Node* minus_y = cal.scalar_mult (input_one, input_y);
    Node* h = cal.add (sigmoid2, minus_y);
    Node* loss = cal.elements_square_sum (h);
  
    cal.graph.build_reverse_graph ();// 构建计算图的转置图

    for (int i = 0; i < 10000; ++i) {
        Tensor result = cal.graph.forward_propagation ();// 前向传播
        if (i % 1000 == 0) {
            cout << "error"; result.display (); cout << endl;// 输出前向传播结果
        }
        cal.graph.back_propagation ();// 进行反向传播
        cal.graph.release_tensor ();// 清除不用的tensor
    }

    cout << "result: w" << endl;
    input_w1 -> output -> display ();
    input_w2 -> output -> display ();

    // 验证结果
    // 设置计算图中的一些节点隐藏, 从而进行结果验证
    vector<Node*> output_list;
    output_list.push_back (sigmoid2);    
    cal.graph.build_subgraph (output_list);

    cout << "模型输出：" << endl;
    cal.graph.forward_propagation ().display ();// 验证结果   
}
