import logging
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from models.diffusion import load_stable_diffusion, generate_image_from_text
from models.clip import load_clip_model, analyze_image
from models.sam import load_sam_model, segment_image
from utils.image_processing import encode_image_to_base64
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load models
try:
    stable_diffusion_model, stable_diffusion_sampler = load_stable_diffusion()
    clip_model, clip_preprocess = load_clip_model()
    sam_predictor = load_sam_model()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

class PromptRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        prompt = request.prompt
        logger.info(f"Received prompt: {prompt}")
        image = generate_image_from_text(prompt, stable_diffusion_model, stable_diffusion_sampler)
        encoded_image = encode_image_to_base64(image)
        clip_result = analyze_image(image, clip_model, clip_preprocess)
        return {
            "generated_image": encoded_image,
            "clip_analysis": {"confidence_scores": clip_result.tolist()}
        }
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        clip_result = analyze_image(image, clip_model, clip_preprocess)
        segmentation_result = segment_image(image, sam_predictor)
        return {
            "clip_analysis": {"confidence_scores": clip_result.tolist()},
            "segmentation": segmentation_result
        }
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
