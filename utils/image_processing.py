from io import BytesIO
import base64
from PIL import Image

def encode_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))
