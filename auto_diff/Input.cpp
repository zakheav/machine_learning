#include "Input.h"
using namespace std;
Input::Input (string name, float input):Node (name) {
    this -> inputs.push_back(input);
    this -> output = input;
}
