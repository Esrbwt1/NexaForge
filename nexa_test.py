import ast
import time
import numpy as np
import tensorflow as tf

# Sample slow code
def slow_code():
    total = 0
    for i in range(1000000):
        if i % 2 == 0:
            total += i
    return total

# Dummy AI model (simplified for your laptop)
def build_ai_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),  # 4 features: loop size, if count, etc.
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 0-1: optimize or not
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Extract features from AST
def extract_features(node):
    features = [0, 0, 0, 0]  # [loop size, has_if, additions, depth]
    if isinstance(node, ast.For):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            features[0] = int(ast.unparse(node.iter.args[0]))  # Loop size
        features[3] = 1  # Depth
    if any(isinstance(n, ast.If) for n in node.body):
        features[1] = 1  # Has if
    if any(isinstance(n, ast.AugAssign) for n in ast.walk(node)):
        features[2] = 1  # Additions
    return features

# Optimizer with AI
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

# Train AI (dummy data for now)
model = build_ai_model()
X_train = np.array([[1000000, 1, 1, 1], [10, 0, 0, 1], [1000, 1, 0, 1]])  # Features
y_train = np.array([1, 0, 1])  # Should optimize?
model.fit(X_train, y_train, epochs=5, verbose=0)

# Test it
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
exec(optimized_code_str.replace("slow_code", "fast_code"))

# Measure
start = time.time()
result = slow_code()
print(f"Slow: {result}, Time: {time.time() - start}")
start = time.time()
result = fast_code()
print(f"Fast: {result}, Time: {time.time() - start}")