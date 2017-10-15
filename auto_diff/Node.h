#ifndef NODE_H_
#define NODE_H_
#include <string>
#include <vector>
class Node {
    public:
        std::string op_name;
        std::vector<float> inputs;
        std::vector<Node*> parents;
        float output;
        float sum_grad;
        virtual void op ();
        virtual void grad_op ();
        Node ();
        Node (std::string name);
        std::string get_name () const;
};
#endif
