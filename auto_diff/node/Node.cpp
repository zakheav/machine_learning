#include "../include/node/Node.h"
using namespace std;
Node::Node () {
    output = 0;
    sum_grad = 0;
}
Node::Node (string name) {
    this -> op_name = name;
    output = 0;
    sum_grad = 0;
}
void Node::op () {
}
void Node::grad_op () {
}
Node::~Node () {
    delete output;
    delete sum_grad;
}
