import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pathlib import Path

# Load the model and image processor
model_name = "yangy50/garbage-classification"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# Prepare a dummy input for the ONNX export (in the required format)
dummy_input = torch.randn(1, 3, 224, 224)  # Assuming the input size is (224, 224)

# Create directory to save the ONNX model
model_dir = Path("onnx_model")
model_dir.mkdir(exist_ok=True)

# Export the model to ONNX format, specifying input/output names
torch.onnx.export(
    model,
    dummy_input,  # Input example for tracing
    model_dir / "vit_model.onnx",  # Output ONNX file
    opset_version=14,  # Use ONNX opset 11 or higher
    input_names=["input"],  # Name the input layer as 'input'
    output_names=["output"],  # Name the output layer as 'output'
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Dynamic batch size
)

print("ViT model successfully exported to ONNX.")
