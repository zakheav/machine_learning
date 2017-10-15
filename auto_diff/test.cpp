#include "Graph.h"
#include <iostream>
#include <vector>
#include "Input.h"
#include "Add.h"
#include "Mult.h"
#include "Node.h"
using namespace std;
int main() {
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
    graph.build_reverse_graph ();
    cout << graph.forward_propagation () << endl;
    graph.back_propagation ();
    cout << graph.node_map["input1"] -> sum_grad << endl;  
    cout << graph.node_map["input2"] -> sum_grad << endl;
    cout << graph.node_map["input3"] -> sum_grad << endl;
}
