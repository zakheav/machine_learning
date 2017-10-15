#include "Graph.h"
#include <iostream>
#include <vector>
#include "Input.h"
#include "Add.h"
#include "Mult.h"
#include "Node.h"
using namespace std;
int main() {
    /*计算图从左到右，input1，input2，input3 输入都是1
    input1
          \
           add1------mult
          /    \    /
    input2      add2
               /
         input3
    */
    // 构建计算图
    Graph graph = Graph ();
    graph.add_node ("", new Input("input1", 1));
    graph.add_node ("", new Input("input2", 1));
    Node* add1 = new Add ("add1");
    graph.add_node ("input1", add1);
    graph.add_node ("input2", add1);
    graph.add_node ("", new Input("input3", 1));
    Node* add2 = new Add ("add2");
    graph.add_node ("input3", add2);
    graph.add_node ("add1", add2);
    Node* mult = new Mult ("mult");
    graph.add_node ("add1", mult);
    graph.add_node ("add2", mult);

    graph.build_reverse_graph ();// 构建计算图的转置图
    cout << "前向传播结果" << graph.forward_propagation () << endl;// 输出前向传播结果
    graph.back_propagation ();// 进行反向传播
    // 输出对于每个input的导数
    cout << "input1的偏导数值 " << graph.node_map["input1"] -> sum_grad << endl;  
    cout << "input2的偏导数值 " << graph.node_map["input2"] -> sum_grad << endl;
    cout << "input3的偏导数值 " << graph.node_map["input3"] -> sum_grad << endl;
}
