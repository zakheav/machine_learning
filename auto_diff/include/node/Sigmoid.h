#ifndef SIGMOID_H_
#define SIGMOID_H_
#include "Node.h"
/*sigmoid节点*/
class Sigmoid: public Node {
    public:
        Sigmoid (std::string name);
        void op ();
        void grad_op ();
};
#endif
