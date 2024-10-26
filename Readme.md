# Text-to-Image API with Stable Diffusion, CLIP, and SAM

This project provides a FastAPI-based web application that generates images from text prompts and performs image analysis using Stable Diffusion, CLIP, and SAM models. This setup is configured to run on GPU-only environments, ideal for Windows users with CUDA-capable GPUs.

## Requirements

- **Python**: 3.9 or above (recommended)
- **pip**: Version 23.0 or earlier
- **git**: Required for cloning repositories

## Setup Instructions


Follow these steps to set up the project on Windows.

## 1. Clone the Repository

Clone the main project repository and the required dependencies.


git clone https://github.com/CompVis/stable-diffusion
git clone https://github.com/openai/CLIP
git clone https://github.com/facebookresearch/segment-anything

## Requirements

- [Git](https://git-scm.com/)
- [Hugging Face Account](https://huggingface.co/join)

## Steps

1. **Log in to Hugging Face and Accept the Terms**

   - Go to the Stable Diffusion v1 model page on Hugging Face: [Stable Diffusion v1 Model](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
   - Read and accept the model terms if prompted.

2. **Download the Model**

   - Open a terminal or command prompt and authenticate with Hugging Face by running:
     ```bash
     huggingface-cli login
     ```
   - Enter your Hugging Face token when prompted. If you don’t have one, you can create it on [your Hugging Face tokens page](https://huggingface.co/settings/tokens).

   - After logging in, download the model using `wget`:
     ```bash
     wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O model.ckpt
     ```

3. **Move and Rename the Model File**

   - Move the downloaded `model.ckpt` file into the directory:
     ```bash
     mv model.ckpt stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
     ```
4. Open the File:

    1. Go to C:\Users\harih\OneDrive\Desktop\Personal_projects\text-image\taming-transformers\taming\data\.
    Open utils.py in a text editor (like VS Code, Notepad++, or any other code editor).
    
    2. Delete Line 11:

        Locate line 11, which should be:
        python
        Copy code
        from torch._six import string_classes
        Delete this line.

    3. Save the File:

        Save the changes and close the editor.

## Directory Structure

After completing these steps, your folder structure should look like this:

## 2.Make sure pip is at version 23.0 or earlier to avoid dependency issues:
pip install pip==23.0

## 3.Creating a virtual environment helps manage dependencies and prevent conflicts:
python -m venv .venv
.venv\Scripts\activate

## 4.With the virtual environment activated, install the project dependencies:
pip install -r requirements.txt

## 5.Taming Transformers is an external library required for image manipulation. You need to clone and install it manually:
git clone https://github.com/CompVis/taming-transformers
cd taming-transformers
pip install -e .
cd ..

## 6.Run:
python -m uvicorn main:app --reload

```bash
