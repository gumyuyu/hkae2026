import base64

def generate_image_from_text(text: str) -> dict:
    fake_image_content = f"Generated image for: {text}".encode("utf-8")
    b64_image = base64.b64encode(fake_image_content).decode("utf-8")
    return {"type": "image", "filename": "image.png", "file": b64_image}

def generate_object_from_image(image_base64: str) -> dict:
    fake_object_content = f"Generated 3D object from image data (truncated: {image_base64[:20]})".encode("utf-8")
    b64_object = base64.b64encode(fake_object_content).decode("utf-8")
    return {"type": "object", "filename": "object.glb", "file": b64_object}

def generate_object_from_text(text: str) -> dict:
    image_result = generate_image_from_text(text)
    object_result = generate_object_from_image(image_result["file"])
    return object_result
