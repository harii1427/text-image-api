import sys
sys.path.append("CLIP")

import clip
import torch
from PIL import Image

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

def analyze_image(image, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits_per_image, _ = model(image_input)
    return logits_per_image.softmax(dim=-1).cpu().numpy()
