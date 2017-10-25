#include "../include/node/Trace.h"
#include <iostream>
using namespace std;
Trace::Trace (string name):Node (name) {
}
void Trace::op () {
    if (parents[0] -> output -> shape[0] != parents[0] -> output -> shape[1]) {
        cout << "not a square matrix" << endl;
    } else {
        output = new Tensor (1, 1);
        for (int i = 0; i < parents[0] -> output -> shape[0]; ++i) {
            output -> tensor[0][0] += parents[0] -> output -> tensor[i][i];
        }
    }
}
void Trace::grad_op () {
    Tensor grad = Tensor (1, parents[0] -> output -> shape[0] * parents[0] -> output -> shape[1]);
    for (int l = 0; l < parents[0] -> output -> shape[0]; ++l) {
        for (int m = 0; m < parents[0] -> output -> shape[1]; ++m) {
            if (l == m) {
                grad.tensor[0][l * parents[0] -> output -> shape[1] + m] = 1;
            }
        }
    }
    if (parents[0] -> sum_grad == 0) {
        if (this -> sum_grad == 0) {
            parents[0] -> sum_grad = new Tensor (grad.tensor);
        } else {
            parents[0] -> sum_grad = this -> sum_grad -> mult (&grad);
        }
    } else {
        if (this -> sum_grad == 0) {
            parents[0] -> sum_grad -> add (&grad, parents[0] -> sum_grad);
        } else {
            parents[0] -> sum_grad -> add (this -> sum_grad -> mult (&grad), parents[0] -> sum_grad);
        }
    }
}
