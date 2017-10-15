#include "Mult.h"
#include <iostream>
using namespace std;
Mult::Mult (string name):Node (name) {
}
void Mult::op () {
    this -> output = this -> inputs[0] * this -> inputs[1];
}
void Mult::grad_op () {
    parents[0] -> sum_grad += parents[1] -> output * sum_grad;
    parents[1] -> sum_grad += parents[0] -> output * sum_grad;
} 
