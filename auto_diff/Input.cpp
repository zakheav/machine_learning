#include "Input.h"
using namespace std;
Input::Input (string name, Tensor* input):Node (name) {
    this -> output = input;
}
