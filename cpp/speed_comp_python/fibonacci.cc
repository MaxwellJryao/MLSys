#include <iostream>
#include <chrono>

int fibonacci_recursive(int n) {
    if (n <= 0)
        return 0;
    else if (n == 1)
        return 1;
    else
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2);
}

int fibonacci_iterative(int n) {
    if (n <= 0)
        return 0;
    else if (n == 1)
        return 1;
    else {
        int a = 0, b = 1;
        for (int i = 2; i <= n; ++i) {
            int temp = b;
            b = a + b;
            a = temp;
        }
        return b;
    }
}

int main() {
    int n = 30;
    auto start_time = std::chrono::high_resolution_clock::now();
    int result_recursive = fibonacci_recursive(n);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> recursive_duration = end_time - start_time;
    std::cout << "Fibonacci recursive result: " << result_recursive << ", Time taken: " << recursive_duration.count() << " seconds" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    int result_iterative = fibonacci_iterative(n);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> iterative_duration = end_time - start_time;
    std::cout << "Fibonacci iterative result: " << result_iterative << ", Time taken: " << iterative_duration.count() << " seconds" << std::endl;

    return 0;
}