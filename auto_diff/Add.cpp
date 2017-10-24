#include "Add.h"
#include <iostream>
using namespace std;
Add::Add (string name):Node (name) {
}
void Add::op () {
    output = parents[0] -> output -> add (parents[1] -> output);
    if (output == 0) {
        cout << "tensor shape not matching in add" << endl;
    }
}
void Add::grad_op () {
    Tensor grad0 = Tensor (output -> shape[0] * output -> shape[1], 
                           parents[0] -> output -> shape[0] * parents[0] -> output -> shape[1]);
    Tensor grad1 = Tensor (output -> shape[0] * output -> shape[1],
                           parents[1] -> output -> shape[0] * parents[1] -> output -> shape[1]);
    // grad0
    for (int i = 0; i < output -> shape[0]; ++i) {
        for (int j = 0; j < output -> shape[1]; ++j) {
            for (int l = 0; l < parents[0] -> output -> shape[0]; ++l) {
                for (int m = 0; m < parents[0] -> output -> shape[1]; ++m) {
                    if (i == l && j == m) {
                        grad0.tensor[i * output -> shape[1] + j][l * parents[0] -> output -> shape[1] + m] = 1;
                    }
                }
            }
        }
    }
    if (parents[0] -> sum_grad == 0) {
        if (this -> sum_grad == 0) {
            parents[0] -> sum_grad = new Tensor (grad0.tensor);
        } else {
            parents[0] -> sum_grad = this -> sum_grad -> mult (&grad0);
        }
    } else {
        if (this -> sum_grad == 0) {
            parents[0] -> sum_grad -> add (&grad0, parents[0] -> sum_grad);
        } else {
            parents[0] -> sum_grad -> add (this -> sum_grad -> mult (&grad0), parents[0] -> sum_grad);
        }
    }
    // grad1
    for (int i = 0; i < output -> shape[0]; ++i) {
        for (int j = 0; j < output -> shape[1]; ++j) {
            for (int l = 0; l < parents[1] -> output -> shape[0]; ++l) {
                for (int m = 0; m < parents[1] -> output -> shape[0]; ++m) {
                    if (i == l && j == m) {
                        grad1.tensor[i * output -> shape[1] + j][l * parents[1] -> output -> shape[1] + m] = 1;
                    }
                }
            }
        }
    }
    if (parents[1] -> sum_grad == 0) {
        if (this -> sum_grad == 0) {
            parents[1] -> sum_grad = new Tensor (grad1.tensor);
        } else {
            parents[1] -> sum_grad = this -> sum_grad -> mult (&grad1);
        }
    } else {
        if (this -> sum_grad == 0) {
            parents[1] -> sum_grad -> add (&grad1, parents[1] -> sum_grad);
        } else {
            parents[1] -> sum_grad -> add (this -> sum_grad -> mult (&grad1), parents[1] -> sum_grad);
        }
    }
}
