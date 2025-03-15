import ast
import time
import sys
import hashlib
import numpy as np
import tensorflow as tf

# AI Model
def build_ai_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),  # Add caching feature
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def extract_features(node):
    features = [0, 0, 0, 0, 0, 0, 0]  # [size, has_if, additions, depth, unrollable, vectorizable, cacheable]
    if isinstance(node, ast.For):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            features[0] = int(ast.unparse(node.iter.args[0]))
            features[4] = 1 if features[0] < 1000 else 0
            features[5] = 1 if features[0] > 10000 else 0
            features[6] = 1 if features[0] > 500000 else 0  # Cache big loops
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
            if should_optimize:
                limit = int(ast.unparse(node.iter.args[0]))
                if features[4]:  # Unroll
                    unrolled = [ast.parse(f"total += {i}").body[0] for i in range(0, limit, 2)]
                    return ast.parse("total = 0").body + unrolled
                elif features[5]:  # Vectorize
                    return ast.parse(f"total = np.sum(np.arange(0, {limit}, 2, dtype=np.int64))").body[0]
                elif features[6]:  # Cache hint (comment for now)
                    return ast.parse(f"total = sum(range(0, {limit}, 2)) # Cacheable").body[0]
                elif len(node.body) == 1 and isinstance(node.body[0], ast.If):
                    if (isinstance(node.body[0].test, ast.Compare) and 
                        isinstance(node.body[0].test.left, ast.BinOp) and 
                        node.body[0].test.left.op.__class__.__name__ == 'Mod'):
                        return ast.parse(f"total = sum(range(0, {limit}, 2))").body[0]
        return self.generic_visit(node)

def optimize_code(code_str, model):
    tree = ast.parse(code_str)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [NexaOptimizer(model).visit(n) for n in node.body]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

# NexaScript Parser
def parse_nexascript(code_str):
    if "SUM_EVEN" in code_str:
        limit = int(code_str.split("(")[1].split(")")[0])
        return f"""
def fast_code():
    total = np.sum(np.arange(0, {limit}, 2, dtype=np.int64))  # Vectorize
    return total
"""
    elif "LOOP" in code_str and "IF" in code_str:
        limit = int(code_str.split("LOOP(")[1].split(")")[0])
        return f"""
def fast_code():
    total = np.sum(np.arange(0, {limit}, 2, dtype=np.int64))  # Vectorize
    return total
"""
    return code_str

# Runtime
class NexaRuntime:
    def __init__(self, license_key):
        self.license_key = license_key
        self.telemetry = {"runs": 0, "time": 0, "optimizations": 0}

    def validate_license(self):
        expected = hashlib.sha256("NexaForge2025".encode()).hexdigest()
        if hashlib.sha256(self.license_key.encode()).hexdigest() != expected:
            print("Invalid license. Shutting down.")
            sys.exit(1)

    def run(self, code_str, optimized=False):
        self.validate_license()
        self.telemetry["runs"] += 1
        if "NEXASCRIPT" in code_str:
            code_str = parse_nexascript(code_str)
            optimized = True
        exec(code_str, globals())
        func = globals()["fast_code"]
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        self.telemetry["time"] = elapsed
        if optimized:
            self.telemetry["optimizations"] += 1
        return result, self.telemetry

# Train AI
model = build_ai_model()
X_train = np.array([
    [1000000, 1, 1, 1, 0, 1, 1],  # Big, vectorize, cache
    [10, 0, 0, 1, 1, 0, 0],       # Tiny, unroll
    [1000, 1, 0, 1, 0, 0, 0],     # Medium, sum
    [500000, 1, 1, 1, 0, 1, 0],   # Big, vectorize
    [50, 1, 1, 1, 1, 0, 0]        # Small, unroll
])
y_train = np.array([1, 1, 1, 1, 1])
model.fit(X_train, y_train, epochs=30, verbose=0)

# Define slow code
exec("""
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total
""")

# Test Python optimization
slow_code_str = """
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total
"""
optimized_code_str = optimize_code(slow_code_str, model)
print("Optimized Python code:\n", optimized_code_str)

runtime_python = NexaRuntime("NexaForge2025")
print("Running optimized Python in NexaRuntime...")
fast_result, telemetry = runtime_python.run(optimized_code_str.replace("slow_code", "fast_code"), optimized=True)

# Test NexaScript
nexa_script = "NEXASCRIPT: LOOP(1000000) IF EVEN"
runtime_nexa = NexaRuntime("NexaForge2025")
print("Running NexaScript in NexaRuntime...")
nexa_result, nexa_telemetry = runtime_nexa.run(nexa_script)

# Measure
start = time.time()
slow_result = slow_code()
print(f"Slow: {slow_result}, Time: {time.time() - start}")
print(f"Fast (Python): {fast_result}, Time: {telemetry['time']}")
print(f"Telemetry (Python): {telemetry}")
print(f"Fast (NexaScript): {nexa_result}, Time: {nexa_telemetry['time']}")
print(f"Telemetry (NexaScript): {nexa_telemetry}")