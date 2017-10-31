#ifndef CAL_H_
#define CAL_H_
#include "../include/Graph.h"
#include "../include/node/Node.h"
class Calculator {
    private:
        int node_id;
    public:
        Graph graph;
        Calculator ();
        Node* add (Node* a, Node* b);
        Node* mult (Node* a, Node* b);
        Node* scalar_mult (Node* scalar, Node* b);
        Node* sigmoid (Node* a);
        Node* elements_square_sum (Node* a);
        Node* trace (Node* a);
        Node* input_variable (Tensor* tensor);//会自动初始化
        Node* input_data (Tensor* tensor);
};
#endif
