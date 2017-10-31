#include "include/Calculator.h"
#include <string>
#include <sstream>
#include <stdlib.h>
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
Calculator::Calculator () {
    this -> node_id = 0;
    this -> graph = Graph ();
}
Node* Calculator::add (Node* a, Node* b) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* add = new Add ("add" + oss.str ());
    graph.add_node (a -> op_name, add);
    graph.add_node (b -> op_name, add);
    return add;
}
Node* Calculator::mult (Node* a, Node* b) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* mult = new Mult ("mult" + oss.str ());
    graph.add_node (a -> op_name, mult);
    graph.add_node (b -> op_name, mult);
    return mult;
}
Node* Calculator::scalar_mult (Node* scalar, Node* b) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* scalar_mult = new ScalarMult ("scalar_mult" + oss.str ());
    graph.add_node (scalar -> op_name, scalar_mult);
    graph.add_node (b -> op_name, scalar_mult);
    return scalar_mult;
}
Node* Calculator::sigmoid (Node* a) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* sigmoid = new Sigmoid ("sigmoid" + oss.str ());
    graph.add_node (a -> op_name, sigmoid);
    return sigmoid;
}
Node* Calculator::elements_square_sum (Node* a) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* ess = new SquareSum ("square_sum" + oss.str ());
    graph.add_node (a -> op_name, ess);
    return ess;
}
Node* Calculator::trace (Node* a) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* trace = new Trace ("trace" + oss.str ());
    graph.add_node (a -> op_name, trace);
    return trace;
}
Node* Calculator::input_variable (Tensor* tensor) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    // 初始化tensor
    for (int i = 0; i < tensor -> shape[0]; ++i) {
        for (int j = 0; j < tensor -> shape[1]; ++j) {
            tensor -> tensor[i][j] = (rand () % 1000) / 1000.0;
        }
    }
    Node* input_v = new Input ("input_v" + oss.str (), tensor, 1);
    graph.add_node ("", input_v);
    return input_v;
}
Node* Calculator::input_data (Tensor* tensor) {
    ostringstream oss;
    ++node_id;
    oss << "" << node_id;
    Node* input_d = new Input ("input_d" + oss.str (), tensor, 0);
    graph.add_node ("", input_d);
    return input_d;
}
