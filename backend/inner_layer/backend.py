import base64
from inner_layer.models.zimage.zimage import generate_image_base64
from inner_layer.models.hunyuan3d.hunyuan3d import generate_3d_object_from_image_base64

def generate_image_from_text(text: str) -> str:
    return generate_image_base64(text) 

def generate_object_from_image(image_base64: str) -> str:
    return generate_3d_object_from_image_base64(image_base64)

def generate_object_from_text(text: str) -> str:
    """Pipeline: text → image → 3D model."""
    image_result = generate_image_base64(text)
    object_result = generate_3d_object_from_image_base64(image_result)
    return object_result
