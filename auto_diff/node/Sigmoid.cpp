#include "../include/node/Sigmoid.h"
#include <cmath>
using namespace std;
Sigmoid::Sigmoid (string name):Node (name) {
}
void Sigmoid::op () {
    output = new Tensor (parents[0] -> output -> shape[0], parents[0] -> output -> shape[1]);
    for (int i = 0; i < output -> shape[0]; ++i) {
        for (int j = 0; j < output -> shape[1]; ++j) {
            output -> tensor[i][j] = 1.0 / (1 + pow (2.718, 0 - parents[0] -> output -> tensor[i][j]));
        }
    }
}
void Sigmoid::grad_op () {
    Tensor grad = Tensor (output -> shape[0] * output -> shape[1], 
                          parents[0] -> output -> shape[0] * parents[0] -> output -> shape[1]);
    for (int i = 0; i < output -> shape[0]; ++i) {
        for (int j = 0; j < output -> shape[1]; ++j) {
            for (int l = 0; l < parents[0] -> output -> shape[0]; ++l) {
                for (int m = 0; m < parents[0] -> output -> shape[1]; ++m) {
                    if (i == l && j == m) {
                        grad.tensor[i * output -> shape[1] + j][l * parents[0] -> output -> shape[1] + m] = 
                        output -> tensor[i][j] * (1 - output -> tensor[i][j]);
                    }
                }
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
