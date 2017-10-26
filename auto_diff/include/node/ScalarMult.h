#ifndef SCALARMULT_H_
#define SCALARMULT_H_
#include "Node.h"
class ScalarMult: public Node {
    public:
        ScalarMult (std::string name);
        void op ();
        void grad_op ();
};
#endif
