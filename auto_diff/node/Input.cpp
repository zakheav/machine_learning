#include "../include/node/Input.h"
using namespace std;
Input::Input (string name, Tensor* input, int need_update):Node (name) {
    this -> output = input;
    this -> need_update = need_update;
}
