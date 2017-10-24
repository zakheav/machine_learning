#ifndef INPUT_H_
#define INPUT_H_
#include "Node.h"
/*输入节点*/
class Input: public Node {
    public:
        Input (std::string name, Tensor* input);
};
#endif
