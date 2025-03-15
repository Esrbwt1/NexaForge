import ast  # Python’s built-in code parser
import time  # To measure speed gains

# Sample slow code: A dumb loop adding numbers
def slow_loop():
    total = 0
    for i in range(1000000):  # 1 million iterations
        total += i
    return total

# Optimizer: Turns slow code into fast code
def optimize_code(code_str):
    tree = ast.parse(code_str)  # Parse the code into a tree structure
    # Dummy optimization: Replace loop with Python’s built-in sum()
    new_code = "def fast_loop():\n    return sum(range(1000000))"
    return new_code

# Test both versions
slow_code = "def slow_loop():\n    total = 0\n    for i in range(1000000):\n        total += i\n    return total"
fast_code = optimize_code(slow_code)  # Generate the optimized version
exec(fast_code)  # Run the new code to define fast_loop()

# Measure slow version
start = time.time()
result = slow_loop()
print(f"Slow: {result}, Time: {time.time() - start}")

# Measure fast version
start = time.time()
result = fast_loop()
print(f"Fast: {result}, Time: {time.time() - start}")