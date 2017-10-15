#include "Node.h"
using namespace std;
Node::Node () {
    this -> output = 0.0;
}
Node::Node (string name) {
    this -> sum_grad = 0.0;
    this -> output = 0.0;
    this -> op_name = name;
}
void Node::op () {
}
void Node::grad_op () {
}
