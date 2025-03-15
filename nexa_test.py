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
        expected = hashlib.sha256("NexaForge2025".encode()).hexdigest()
        if hashlib.sha256(self.license_key.encode()).hexdigest() != expected:
            print("Invalid license. Shutting down.")
            sys.exit(1)

    def run(self, code_str):
        self.validate_license()
        self.telemetry["runs"] += 1
        start = time.time()
        if "NEXASCRIPT" in code_str:  # Stub for future custom language
            print("NexaScript detected—coming soon!")
            exec("def fast_code(): return 249999500000")  # Placeholder
        else:
            exec(code_str, globals())
        elapsed = time.time() - start
        self.telemetry["time"] += elapsed
        return self.telemetry

# Optimized code
optimized_code_str = """
def fast_code():
    total = sum(range(0, 1000000, 2))
    return total
"""

# Slow code
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total

# Test runtime
runtime = NexaRuntime("NexaForge2025")
print("Running optimized code in NexaRuntime...")
telemetry = runtime.run(optimized_code_str)

# Measure
start = time.time()
slow_result = slow_code()
print(f"Slow: {slow_result}, Time: {time.time() - start}")
start = time.time()
fast_result = fast_code()
print(f"Fast: {fast_result}, Time: {time.time() - start}")
print(f"Telemetry: {telemetry}")

# Test NexaScript stub
nexa_script = "NEXASCRIPT: SUM_EVEN(1000000)"
runtime_nexa = NexaRuntime("NexaForge2025")
telemetry_nexa = runtime_nexa.run(nexa_script)

# Test invalid license
runtime_bad = NexaRuntime("WrongKey")
telemetry_bad = runtime_bad.run(optimized_code_str)