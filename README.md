# EcoSort-Backend

### Description
This backend API leverages a **pre-trained garbage sorting Vision Transformer (ViT) model** to classify various types of waste (e.g., metal, plastic, cardboard) from an image. The API provides an endpoint that allows the front-end to make requests for image classification, returning the predicted class labels and their associated probabilities.

### Installation and Setup

#### 1. Clone the Repository
Start by cloning this repository to your local machine:
```bash
git clone https://github.com/your-username/EcoSort-Backend.git
cd EcoSort-Backend
```
#### 2. Create a Virtual Environment
It is recommended to create a virtual environment to manage your dependencies. You can do this using Python's built-in ```venv``` module:
```bash
python -m venv openvino_env
source openvino_env/bin/activate  # On Windows: openvino_env\Scripts\activate
```
#### 3. Install Dependencies
Once inside the virtual environment, install all necessary dependencies using ```pip```:
```bash
pip install fastapi uvicorn openvino onnxruntime numpy pillow
```
#### 4. Run the API
After installing the dependencies, you can start the FastAPI server with Uvicorn. Use the following command to run the API on your local machine:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
### Front-End Integration
Once the API server is running, you can launch the front-end from this repository:
[EcoSort Front-End](https://github.com/Amankhan2004/EcoSort)

The front-end allows users to take a picture of a disposable item, which will be sent to the API for classification. The API will return the material of the item and whether it is recyclable or not.

### Usage
Once the API server is running, you can send POST requests with image files to the /predict/ endpoint. The API will process the image and return a JSON response with the predicted class labels and their probabilities.

### Example Request
Send a POST request using curl:
```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpg'
```
### Example Response
The response will look like this:
```bash
[
    {"label": "metal", "score": 0.8635},
    {"label": "trash", "score": 0.8072},
    {"label": "plastic", "score": 0.6625}
]
```
### Notes
- **Activate Virtual Environment**: Make sure to activate your virtual environment each time before running the server.
- **Port Configuration**: If you're deploying this API on a cloud platform or a local network, make sure the appropriate ports are open.
