import torch
import torch.nn.functional as F
from fastai.vision.all import *
from fastapi import FastAPI, File, UploadFile
import io
import numpy as np
from PIL import Image
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Try to load the model and handle the case when the model is missing
model = None
# Load models
resnet_model = load_learner('resnet50_model.pkl')
vgg_model = load_learner('vgg16_model.pkl')

def preprocess_image(file: UploadFile, size=(224, 224)):
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()  # CHW format
    return img_tensor.unsqueeze(0)  # Add batch dimension

def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model.model(img_tensor.to(model.dls.device))
    return predictions

def get_prediction_details(model, predictions):
    probs = F.softmax(predictions, dim=1)
    conf, predicted_idx = torch.max(probs, 1)
    predicted_label = model.dls.vocab[predicted_idx.item()]
    
    class_probs = {model.dls.vocab[i]: prob.item() for i, prob in enumerate(probs[0])}
    
    return {
        "predicted_label": predicted_label,
        "confidence": conf.item(),
        "class_probabilities": class_probs
    }

@app.post("/predict/resnet")
async def predict_resnet(file: UploadFile = File(...)):
    img_tensor = preprocess_image(file)
    predictions = predict(resnet_model, img_tensor)
    result = get_prediction_details(resnet_model, predictions)
    return result

@app.post("/predict/vgg")
async def predict_vgg(file: UploadFile = File(...)):
    img_tensor = preprocess_image(file)
    predictions = predict(vgg_model, img_tensor)
    result = get_prediction_details(vgg_model, predictions)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Blood Cell Classification API. Use /predict/ for predictions."}