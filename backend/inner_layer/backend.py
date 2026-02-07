import base64
from email.mime import image
from inner_layer.models.zimage.zimage import generate_image_base64
from inner_layer.models.hunyuan3D.hunyuan3d import generate_3d_object_from_image_base64

def generate_image_from_text(text: str) -> str:
    return generate_image_base64(text, convert_to_glb=True)

def generate_object_from_image(image_base64: str) -> str:
    return generate_3d_object_from_image_base64(image_base64)

def generate_object_from_text(text: str) -> str:
    image_path = generate_image_base64(text, convert_to_glb=False)
    object_result = generate_3d_object_from_image_base64(image_path)
    return object_result
