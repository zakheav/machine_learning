#ifndef GRAPH_H_
#define GRAPH_H_
#include <unordered_map>
#include <vector>
#include <queue>
#include "Node.h"
class Graph {
    public:
        std::unordered_map<std::string, Node*> node_map;
        std::unordered_map<std::string, std::vector<Node*> > adj_table;
        std::unordered_map<std::string, std::vector<Node*> > reverse_table;
        void add_node (std::string parent_name, Node* node); 
        void topological_sort (std::unordered_map<std::string, std::vector<Node*> > &adj_table, std::vector<Node*> &result);
        void build_reverse_graph ();
        float forward_propagation ();
        void back_propagation ();
        std::string to_string ();
        ~Graph ();
};
#endif
