import time

def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


def fibonacci_iterative(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

def main():
    n = 30
    start_time = time.time()
    result = fibonacci_recursive(n)
    end_time = time.time()
    print(f"Fibonacci recursive result: {result}, Time taken: {end_time - start_time} seconds")
    
    start_time = time.time()
    result = fibonacci_iterative(n)
    end_time = time.time()
    print(f"Fibonacci iterative result: {result}, Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
