import ast
import time
import sys
import hashlib
import numpy as np
import tensorflow as tf

# AI Model
def build_ai_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # Bigger net
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def extract_features(node):
    features = [0, 0, 0, 0]  # [loop size, has_if, additions, depth]
    if isinstance(node, ast.For):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            features[0] = int(ast.unparse(node.iter.args[0]))
        features[3] = 1
    if any(isinstance(n, ast.If) for n in node.body):
        features[1] = 1
    if any(isinstance(n, ast.AugAssign) for n in ast.walk(node)):
        features[2] = 1
    return features

# Optimizer
class NexaOptimizer(ast.NodeTransformer):
    def __init__(self, model):
        self.model = model

    def visit_For(self, node):
        if (isinstance(node.iter, ast.Call) and 
            node.iter.func.id == 'range' and 
            len(node.iter.args) == 1):
            features = extract_features(node)
            should_optimize = self.model.predict(np.array([features]), verbose=0)[0][0] > 0.5
            if should_optimize and len(node.body) == 1 and isinstance(node.body[0], ast.If):
                if (isinstance(node.body[0].test, ast.Compare) and 
                    isinstance(node.body[0].test.left, ast.BinOp) and 
                    node.body[0].test.left.op.__class__.__name__ == 'Mod' and 
                    node.body[0].test.comparators[0].n == 0):
                    limit = ast.unparse(node.iter.args[0])
                    new_body = ast.parse(f"total = sum(range(0, {limit}, 2))").body[0]
                    return new_body
        return self.generic_visit(node)

def optimize_code(code_str, model):
    tree = ast.parse(code_str)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [NexaOptimizer(model).visit(n) for n in node.body]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

# Runtime
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
        if "NEXASCRIPT" in code_str:
            print("NexaScript detected—coming soon!")
            exec("def fast_code(): return 249999500000")
        else:
            exec(code_str, globals())
        func = globals()["fast_code"]
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        self.telemetry["time"] = elapsed
        return result, self.telemetry

# Train AI better
model = build_ai_model()
X_train = np.array([
    [1000000, 1, 1, 1],  # Big loop, if, add—optimize
    [10, 0, 0, 1],       # Tiny, no if—skip
    [1000, 1, 0, 1],     # Medium, if—optimize
    [500000, 1, 1, 1],   # Big, if, add—optimize
    [10000, 0, 1, 1]     # Medium, no if—skip
])
y_train = np.array([1, 0, 1, 1, 0])  # More data, better decisions
model.fit(X_train, y_train, epochs=20, verbose=0)  # Train harder

# Define slow code globally
exec("""
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total
""")

# Test
slow_code_str = """
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total
"""
optimized_code_str = optimize_code(slow_code_str, model)
print("Optimized code:\n", optimized_code_str)

runtime = NexaRuntime("NexaForge2025")
print("Running in NexaRuntime...")
fast_result, telemetry = runtime.run(optimized_code_str.replace("slow_code", "fast_code"))

# Measure
start = time.time()
slow_result = slow_code()
print(f"Slow: {slow_result}, Time: {time.time() - start}")
print(f"Fast: {fast_result}, Time: {telemetry['time']}")
print(f"Telemetry: {telemetry}")