#ifndef ADD_H_ 
#define ADD_H_ 
#include "Node.h"
/*加法节点*/
class Add: public Node {
    public:
        Add (std::string name);
        void op ();
        void grad_op ();
};
#endif
