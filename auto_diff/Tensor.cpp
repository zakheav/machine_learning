#include <iostream>
#include "Tensor.h"
using namespace std;
Tensor::Tensor () {
}
Tensor::Tensor (int row, int col) {
    shape[0] = row;
    shape[1] = col;
    tensor.resize (row);
    for (int i = 0; i < row; ++i) {
        tensor[i].resize (col);
    }
}
Tensor::Tensor (vector<vector<float> > &data) {
    shape[0] = data.size();
    shape[1] = data[0].size();
    tensor = data;
}
Tensor* Tensor::mult (Tensor* tensor) {
    Tensor* result = 0;
    if (this -> shape[1] == tensor -> shape[0]) {
        result = new Tensor (this -> shape[0], tensor -> shape[1]);
        for (int i = 0; i < this -> shape[0]; ++i) {
            for (int j = 0; j < tensor -> shape[1]; ++j) {
                for (int k = 0; k < this -> shape[1]; ++k) {
                    result -> tensor[i][j] += this -> tensor[i][k] * tensor -> tensor[k][j];
                }
            }
        }
    }
    return result;
}
void Tensor::add (Tensor* tensor, Tensor* result) {
    for (int i = 0; i < this -> shape[0]; ++i) {
        for (int j = 0; j < this -> shape[1]; ++j) {
            result -> tensor[i][j] = this -> tensor[i][j] + tensor -> tensor[i][j];
        }
    }
}
Tensor* Tensor::add (Tensor* tensor) {
    Tensor* result = 0;
    if (this -> shape[0] == tensor -> shape[0] && this -> shape[1] == tensor -> shape[1]) {
        result = new Tensor (this -> shape[0], this -> shape[1]);
        for (int i = 0; i < this -> shape[0]; ++i) {
            for (int j = 0; j < this -> shape[1]; ++j) {
                result -> tensor[i][j] = this -> tensor[i][j] + tensor -> tensor[i][j];
            }
        }
    }
    return result;
}
void Tensor::display () {
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            cout << tensor[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
