#include "Add.h"
using namespace std;
Add::Add (string name):Node (name) {
}
void Add::op () {
    this -> output = this -> inputs[0] + this -> inputs[1];
}
void Add::grad_op () {
    parents[0] -> sum_grad += sum_grad * 1.0;
    parents[1] -> sum_grad += sum_grad * 1.0;
}
