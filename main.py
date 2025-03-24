import torch
import torch.nn.functional as F
from fastai.vision.all import *
from fastapi import FastAPI, File, UploadFile, HTTPException
import io
import numpy as np
from PIL import Image
import logging
import os
import sys
import pickle
from huggingface_hub import hf_hub_download
from pathlib import PosixPath

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom unpickler to handle WindowsPath and persistent IDs
class FixedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Replace WindowsPath with PosixPath
        if module == "pathlib" and name == "WindowsPath":
            return PosixPath
        return super().find_class(module, name)
    
    def persistent_load(self, pid):
        # Handle persistent IDs - often used in FastAI models
        # This is a simple implementation that might need refinement
        logger.info(f"Handling persistent_load for: {pid}")
        if isinstance(pid, tuple) and len(pid) > 1:
            # Common format is (tag, key)
            tag, key = pid[0], pid[1]
            if tag == 'tensortype':
                # For tensor types, return the corresponding tensor type
                return getattr(torch, key)
            elif tag == 'storage':
                # For storage, create an empty tensor storage
                storage_type, key, location, size = pid[1:]
                return torch.storage._TypedStorage()
            elif tag == 'module':
                # For modules, import the module
                return __import__(key, fromlist=[''])
        # Default: return the pid itself (may or may not work)
        return pid

# Initialize FastAPI
app = FastAPI()

# Define the repository details on Hugging Face
REPO_ID = "RoadHaus/BC_Classification"

# Load the models from Hugging Face with proper error handling
def load_model_from_huggingface(model_filename):
    try:
        logger.info(f"Downloading model {model_filename} from Hugging Face")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=model_filename)
        logger.info(f"Model downloaded to {model_path}")
        
        # First try using torch.load directly with custom pickle module
        try:
            # Try loading with the custom unpickler
            with open(model_path, 'rb') as f:
                unpickler = FixedUnpickler(f)
                unpickler.persistent_load = unpickler.persistent_load  # Enable persistent_load
                return unpickler.load()
        except Exception as e:
            logger.error(f"Custom unpickler failed: {str(e)}")
            
            # Fallback to loading directly with torch
            try:
                # Use a more direct approach with torch.load
                return torch.load(
                    model_path, 
                    map_location=torch.device('cpu'),
                    pickle_module=pickle,
                    weights_only=False  # Try to load the full model
                )
            except Exception as e:
                logger.error(f"torch.load fallback failed: {str(e)}")
                
                # Last resort: try loading with fastai's load_learner
                try:
                    from fastai.learner import load_learner
                    return load_learner(model_path)
                except Exception as e:
                    logger.error(f"fastai load_learner fallback failed: {str(e)}")
                    raise
                
    except Exception as e:
        logger.error(f"Error loading model {model_filename}: {str(e)}")
        raise

# Global variables for models
resnet_model = None
vgg_model = None

# Load models on startup
@app.on_event("startup")
async def startup_event():
    global resnet_model, vgg_model
    try:
        logger.info("Loading ResNet model")
        resnet_model = load_model_from_huggingface("resnet50_model.pkl")
        logger.info("ResNet model loaded successfully")
        
        logger.info("Loading VGG model")
        vgg_model = load_model_from_huggingface("vgg16_model.pkl")
        logger.info("VGG model loaded successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Continue running even if model loading fails, so health checks can work

def preprocess_image(file: UploadFile, size=(224, 224)):
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize(size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()  # CHW format
        return img_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def predict(model, img_tensor):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
    try:
        if resnet_model is None:
            raise HTTPException(status_code=503, detail="ResNet model not loaded")
        
        img_tensor = preprocess_image(file)
        predictions = predict(resnet_model, img_tensor)
        result = get_prediction_details(resnet_model, predictions)
        return result
    except Exception as e:
        logger.error(f"Error in ResNet prediction: {str(e)}")
        return {"error": str(e), "traceback": str(sys.exc_info())}

@app.post("/predict/vgg")
async def predict_vgg(file: UploadFile = File(...)):
    try:
        if vgg_model is None:
            raise HTTPException(status_code=503, detail="VGG model not loaded")
        
        img_tensor = preprocess_image(file)
        predictions = predict(vgg_model, img_tensor)
        result = get_prediction_details(vgg_model, predictions)
        return result
    except Exception as e:
        logger.error(f"Error in VGG prediction: {str(e)}")
        return {"error": str(e), "traceback": str(sys.exc_info())}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "resnet": "loaded" if resnet_model is not None else "not loaded",
            "vgg": "loaded" if vgg_model is not None else "not loaded"
        }
    }

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Blood Cell Classification API. Use /predict/resnet or /predict/vgg for predictions."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)