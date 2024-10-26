import sys
sys.path.append("segment-anything")

from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
from PIL import Image

def load_sam_model():
    sam_model = sam_model_registry["default"]().to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam_model)
    return predictor

def segment_image(image, predictor):
    image_np = np.array(image)
    predictor.set_image(image_np)
    masks, _, _ = predictor.predict()
    return masks
