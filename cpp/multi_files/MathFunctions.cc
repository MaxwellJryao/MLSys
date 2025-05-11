double power(double base, int exponent) {
    if (exponent == 0) {
        return 1;
    }

    if (exponent < 0) {
        return 1 / power(base, -exponent);
    }
    
    double result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}