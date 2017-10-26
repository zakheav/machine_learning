#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>
class Tensor {
    public:
        std::vector<std::vector<float> > tensor;
	int shape[2];
        Tensor ();
	Tensor (int row, int col);
        Tensor (std::vector<std::vector<float> > &data);
        Tensor* mult (Tensor* tensor);
        Tensor* scalar_mult (float a);
        void add (Tensor* tensor, Tensor* result);
        Tensor* add (Tensor* tensor);
        void display ();
};
#endif
