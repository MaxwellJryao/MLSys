#include <iostream>
#include "MathFunctions.h"

int main() {
    double base = 2;
    int exponent = 3;
    double result = power(base, exponent);
    std::cout << "power(" << base << ", " << exponent << ") = " << result << std::endl;
    return 0;
}