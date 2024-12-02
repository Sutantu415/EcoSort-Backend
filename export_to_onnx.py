import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load the model and feature extractor
model_name = "yangy50/garbage-classification"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX with opset version 14
onnx_path = "garbage_classification.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14,  
)

print(f"Model exported to {onnx_path}")
