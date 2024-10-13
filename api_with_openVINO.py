from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openvino.runtime import Core
from PIL import Image
import numpy as np
import io
import torch.nn.functional as F
import torch

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenVINO runtime and load the compiled model on GPU
core = Core()
model_path = "openvino_model/vit_model.xml"  # Path to your OpenVINO model files
compiled_model = core.compile_model(model=model_path, device_name="GPU")  # Use GPU for inference

# Define input/output layers
input_layer = compiled_model.input("input")
output_layer = compiled_model.output("output")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess the image: resize, convert to NumPy array, and reshape
    image = image.resize((224, 224))  # Resize to 224x224 for ViT model input
    image = np.array(image).astype(np.float32).transpose(2, 0, 1)  # Convert to NCHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Run inference using OpenVINO
    logits = compiled_model([image])[output_layer]
    
    # Log the logits (for comparison purposes)
    print("Logits from OpenVINO:", logits)

    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()

    # Convert probabilities to list and ensure they're Python floats
    probabilities_list = [float(prob) for prob in probabilities[0]]

    # Map probabilities to class labels and sort by score (descending)
    labels = ["metal", "trash", "plastic", "cardboard", "paper"]  # Your predefined class labels
    labeled_results = [{"label": label, "score": round(prob, 4)} for label, prob in zip(labels, probabilities_list)]
    labeled_results = sorted(labeled_results, key=lambda x: x["score"], reverse=True)

    return labeled_results

@app.get("/")
async def root():
    return {"message": "OpenVINO inference with ViT model on GPU"}
