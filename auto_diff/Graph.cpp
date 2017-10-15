#include <iostream>
#include "Graph.h"
using namespace std;
void Graph::add_node (string parent_name, Node* node) {
    node_map.insert(make_pair(node -> op_name, node));
    if (parent_name != "") {
        adj_table[parent_name].push_back(node);
        Node* parent_node = node_map[parent_name];
        (node -> parents).push_back(parent_node);
    }
}
float Graph::forward_propagation () {
    vector<Node*> topo_result;
    topological_sort (adj_table, topo_result);
    vector<Node*>::iterator vec_it = topo_result.begin ();
    float result = 0.0;
    while (vec_it != topo_result.end ()) {
        string node_name = (*vec_it) -> op_name;
        if (reverse_table.find (node_name) != reverse_table.end()) {// parents nodes exist
            vector<Node*> parents = reverse_table[node_name];
            vector<Node*>::iterator parents_it = parents.begin ();
            while (parents_it != parents.end ()) {// copy parents output to child's inputs
                (*vec_it) -> inputs.push_back ((*parents_it) -> output);
                ++parents_it;
            }
            (*vec_it) -> op ();
        }
        result = (*vec_it) -> output;
        ++vec_it;
    }
    return result;
}
void Graph::back_propagation () {
    vector<Node*> topo_result;
    topological_sort (reverse_table, topo_result);
    topo_result[0] -> sum_grad = 1.0;
    vector<Node*>::iterator vec_it = topo_result.begin ();
    while (vec_it != topo_result.end ()) {
        cout << (*vec_it) -> op_name << endl;
        (*vec_it) -> grad_op ();
        ++vec_it;
    }
}
void Graph::build_reverse_graph () {
    unordered_map<string, vector<Node*> >::iterator adj_table_it = adj_table.begin();
    while (adj_table_it != adj_table.end()) {
        vector<Node*>::iterator vec_it = (adj_table_it -> second).begin();
        Node* reverse_child_node = node_map[adj_table_it -> first];
        while (vec_it != (adj_table_it -> second).end()) {
            string reverse_parent_name = (*vec_it) -> op_name;
            reverse_table[reverse_parent_name].push_back(reverse_child_node);
            ++vec_it;
        } 
        ++adj_table_it;
    }
}
void Graph::topological_sort (unordered_map<string, vector<Node*> > &adj_table, vector<Node*> &result) {
    unordered_map<string, int> indegree;
    unordered_map<string, Node*>::iterator node_map_it = node_map.begin();
    while (node_map_it != node_map.end()) {
        indegree.insert(make_pair(node_map_it -> first, 0));
        ++node_map_it;
    }
    unordered_map<string, vector<Node*> >::iterator adj_table_it = adj_table.begin();
    while (adj_table_it != adj_table.end()) {
        vector<Node*>::iterator vec_it = (adj_table_it -> second).begin();
        while (vec_it != (adj_table_it -> second).end()) {
            ++indegree[(*vec_it) -> op_name];
            ++vec_it;
        }
        ++adj_table_it; 
    }
    queue<Node*> q;
    unordered_map<string, int>::iterator indegree_it = indegree.begin();
    while (indegree_it != indegree.end()) {
        if (indegree_it -> second == 0) {
            q.push(node_map[indegree_it -> first]);
        }
        ++indegree_it;
    }
    while (!q.empty()) {
        Node* node = q.front();
        q.pop();
        result.push_back(node);
        vector<Node*>::iterator vec_it = adj_table[node -> op_name].begin();
        while (vec_it != adj_table[node -> op_name].end()) {
            --indegree[(*vec_it) -> op_name];
            if (indegree[(*vec_it) -> op_name] == 0) {
                q.push(*vec_it);
            }
            ++vec_it;
        }
    }
}
string Graph::to_string () {
    string s = "";
    unordered_map<string, vector<Node*> >::iterator map_it = reverse_table.begin();
    while (map_it != reverse_table.end()) {
        s += map_it -> first + ":";
        vector<Node*>::iterator vec_it = reverse_table[map_it -> first].begin();
        while (vec_it != reverse_table[map_it -> first].end()) {
            s += (*vec_it) -> op_name + " ";
            ++vec_it;
        }
        s += '\n';
        ++map_it;
    }
    return s;
}
Graph::~Graph () {
    // free node_map
    cout << "free adj_table" << endl;
    adj_table.clear();

    // free reverse_table
    cout << "free reverse_table" << endl;
    reverse_table.clear();

    // free node_map
    cout << "free node_map" << endl;
    unordered_map<string, Node*>::iterator node_map_it = node_map.begin();
    while (node_map_it != node_map.end()) {
        delete (node_map_it -> second);
        node_map_it -> second = 0;
        ++node_map_it;
    }
    node_map.clear();
}
