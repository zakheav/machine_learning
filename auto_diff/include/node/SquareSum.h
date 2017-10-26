#ifndef SQUARESUM_H_
#define SQUARESUM_H_
#include "Node.h"
class SquareSum: public Node {
    public:
        SquareSum (std::string);
        void op ();
        void grad_op ();
};
#endif
