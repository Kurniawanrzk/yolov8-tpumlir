import onnx
from onnx import helper, shape_inference

# Load the ONNX model
model = onnx.load("workspace/best.onnx")

# Modify or remove unsupported nodes (pseudo-code)
for node in model.graph.node:
    if node.op_type == "ConvInteger":
        node.op_type = "Conv"  # Modify as per compatibility

# Save the updated model
onnx.save(model, "best_fixed.onnx")
