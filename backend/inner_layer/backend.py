import base64
from inner_layer.models.zimage import generate_image_base64

def generate_image_from_text(text: str) -> bytes:
    b64_image = generate_image_base64(text)
    return base64.b64decode(b64_image)

def generate_object_from_image(image_base64: str) -> dict:
    """Stub: Convert image → 3D model (base64-encoded)."""
    fake_object_content = f"Generated 3D object from image data (truncated: {image_base64[:20]})".encode("utf-8")
    b64_object = base64.b64encode(fake_object_content).decode("utf-8")
    return {
        "type": "object",
        "filename": "object.glb",
        "file": b64_object,
    }

def generate_object_from_text(text: str) -> dict:
    """Pipeline: text → image → 3D model."""
    image_result = generate_image_from_text(text)
    object_result = generate_object_from_image(image_result["file"])
    return object_result
