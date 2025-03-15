import ast
import time
import sys
import hashlib

# Runtime class
class NexaRuntime:
    def __init__(self, license_key):
        self.license_key = license_key
        self.telemetry = {"runs": 0, "time": 0}

    def validate_license(self):
        # Dummy check: Hash must match
        expected = hashlib.sha256("NexaForge2025".encode()).hexdigest()
        if hashlib.sha256(self.license_key.encode()).hexdigest() != expected:
            print("Invalid license. Shutting down.")
            sys.exit(1)

    def run(self, code_str):
        self.validate_license()
        self.telemetry["runs"] += 1
        start = time.time()
        exec(code_str, globals())
        self.telemetry["time"] += time.time() - start
        return self.telemetry

# Sample optimized code (from Step 5)
optimized_code_str = """
def fast_code():
    total = sum(range(0, 1000000, 2))
    return total
"""

# Original slow code
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total

# Test runtime
runtime = NexaRuntime("NexaForge2025")  # Valid key
print("Running optimized code in NexaRuntime...")
telemetry = runtime.run(optimized_code_str)

# Measure both
start = time.time()
slow_result = slow_code()
print(f"Slow: {slow_result}, Time: {time.time() - start}")

start = time.time()
fast_result = fast_code()
print(f"Fast: {fast_result}, Time: {time.time() - start}")
print(f"Telemetry: {telemetry}")

# Test invalid license
runtime_bad = NexaRuntime("WrongKey")
telemetry_bad = runtime_bad.run(optimized_code_str)  # Should fail