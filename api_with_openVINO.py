from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openvino.runtime import Core
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the OpenVINO model
core = Core()
model_path = "openvino_model/garbage_classification.xml"  # Path to your model
compiled_model = core.compile_model(model=model_path, device_name="CPU")  # Use "GPU" if supported
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Predefined labels
labels = ["metal", "trash", "plastic", "cardboard", "paper"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224)) 
    image = np.array(image).astype(np.float32).transpose(2, 0, 1) 
    image = np.expand_dims(image, axis=0)

    # Run inference
    results = compiled_model([image])[output_layer]

    # Apply softmax
    probabilities = np.exp(results) / np.sum(np.exp(results), axis=-1, keepdims=True)
    probabilities = probabilities.flatten()

    # Map probabilities to labels
    labeled_results = [{"label": label, "score": float(prob)} for label, prob in zip(labels, probabilities)]
    labeled_results = sorted(labeled_results, key=lambda x: x["score"], reverse=True)

    return labeled_results

@app.get("/")
async def root():
    return {"message": "OpenVINO-powered garbage classification API"}