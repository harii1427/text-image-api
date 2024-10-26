import sys
sys.path.append("stable-diffusion")

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf

def load_stable_diffusion():
    # Load the configuration
    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    
    # Instantiate the model from the configuration
    model = instantiate_from_config(config.model)
    
    # Load model weights to CPU
    model.load_state_dict(torch.load("stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt", map_location="cpu")["state_dict"], strict=False)

    # Move the model to CPU
    model = model.to("cpu")

    # Initialize the sampler
    sampler = DDIMSampler(model)
    
    return model, sampler

def generate_image_from_text(prompt, model, sampler):
    # Generate an image using the prompt
    with torch.no_grad():
        # Prepare the input for the sampler
        conditioning = model.get_learned_conditioning([prompt])
        
        # Set shape to a lower resolution for testing; adjust to (1, 3, 512, 512) for full resolution
        shape = (1, 3, 256, 256)  # Temporary shape for faster CPU processing
        
        # Sample latents
        latents = sampler.sample(batch_size=1, shape=shape, conditioning=conditioning)

        # Decode latents to an image
        image = model.decode_first_stage(latents)

    return image[0]  # Return the first image if batch_size is 1
