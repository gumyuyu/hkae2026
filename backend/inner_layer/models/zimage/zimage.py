import io
import base64
import uuid
from pathlib import Path
from PIL import Image
import numpy as np
import trimesh
from gradio_client import Client

# --- Ensure output directories ---
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# --- Convert PIL image to GLB ---
def image_to_glb(image: Image.Image, thickness: float = 0.001) -> bytes:
    w, h = image.size
    aspect = w / h

    mesh = trimesh.creation.box(extents=(aspect, 1.0, thickness))

    uv = np.zeros((len(mesh.vertices), 2))
    uv[:, 0] = (mesh.vertices[:, 0] / aspect + 0.5)
    uv[:, 1] = (mesh.vertices[:, 1] + 0.5)

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    return mesh.export(file_type="glb")


# --- Global Gradio client ---
_client = None
def get_gradio_client():
    global _client
    if _client is None:
        _client = Client("mrfakename/Z-Image-Turbo")
    return _client


# --- Main image generation function ---
def generate_image_base64(
    prompt: str,
    height: int = 768,
    width: int = 768,
    steps: int = 8,
    seed: int = 5,
    convert_to_glb: bool = False,
    randomize_seed: bool = True,
) -> str:
    """
    Generate an image from text prompt using Gradio Z-Image-Turbo.
    Returns:
      - base64 PNG if convert_to_glb=False
      - base64 GLB if convert_to_glb=True
    """
    client = get_gradio_client()

    # Predict using the Gradio API
    result, used_seed = client.predict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=seed,
        randomize_seed=randomize_seed,
        api_name="/generate_image"
    )

   

    image = Image.open(result)

    ensure_dir("output/images")
    img_name = f"zimage_{uuid.uuid4().hex}.png"
    img_path = Path("output/images") / img_name
    image.save(img_path)
    print(f"[ZImage] Saved image to {img_path}")

    # --- Return PNG base64 ---
    if not convert_to_glb:
        
        return img_path

    # --- Convert to GLB ---
    glb_bytes = image_to_glb(image)
    glb_name = f"zimage_{uuid.uuid4().hex}.glb"
    glb_path = Path("output/images") / glb_name
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)
    print(f"[ZImage] Saved GLB to {glb_path}")

    return base64.b64encode(glb_bytes).decode("utf-8")
