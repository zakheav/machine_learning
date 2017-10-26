#include <iostream>
#include "include/Graph.h"
using namespace std;
void Graph::add_node (string parent_name, Node* node) {
    node_map.insert(make_pair(node -> op_name, node));
    if (parent_name != "") {
        adj_table[parent_name].push_back(node);
        Node* parent_node = node_map[parent_name];
        (node -> parents).push_back(parent_node);
    }
}
Tensor Graph::forward_propagation () {
    vector<Node*> topo_result;
    topological_sort (adj_table, topo_result);
    vector<Node*>::iterator vec_it = topo_result.begin ();
    Tensor* result;
    while (vec_it != topo_result.end ()) {
        string node_name = (*vec_it) -> op_name;
        if (reverse_table.find (node_name) != reverse_table.end()) {// parents nodes exist
            (*vec_it) -> op ();
        }
        result = (*vec_it) -> output;
        ++vec_it;
    }
    return *result;
}
void Graph::back_propagation () {
    vector<Node*> topo_result;
    topological_sort (reverse_table, topo_result);
    vector<Node*>::iterator vec_it = topo_result.begin ();
    while (vec_it != topo_result.end ()) {
        (*vec_it) -> grad_op ();
        ++vec_it;
    }
    // 更新权值
    vec_it = topo_result.begin ();
    while (vec_it != topo_result.end ()) {
        (*vec_it) -> update ();
        ++vec_it;
    }
    // 释放内存
    vec_it = topo_result.begin ();
    while (vec_it != topo_result.end ()) {
        if ((*vec_it) -> need_update == 0) {
            delete (*vec_it) -> output;
            (*vec_it) -> output = 0;
        }
        delete (*vec_it) -> sum_grad;
        (*vec_it) -> sum_grad = 0;
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
    cout << "free node_map" << endl;
    unordered_map<string, Node*>::iterator node_map_it = node_map.begin();
    while (node_map_it != node_map.end()) {
        delete (node_map_it -> second);
        node_map_it -> second = 0;
        ++node_map_it;
    }
}
