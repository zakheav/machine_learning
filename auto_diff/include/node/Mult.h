#ifndef MULT_H_
#define MULT_H_
#include "Node.h"
/*乘法节点*/
class Mult: public Node {
    public:
        Mult (std::string name);
        void op ();
        void grad_op ();
};
#endif
