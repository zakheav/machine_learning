#ifndef TRACE_H_
#define TRACE_H_
#include "Node.h"
class Trace: public Node {
    public:
        Trace (std::string name);
        void op ();
        void grad_op ();
};
#endif
