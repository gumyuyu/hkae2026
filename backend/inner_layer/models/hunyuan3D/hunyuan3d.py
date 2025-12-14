import io
import base64
import time
from pathlib import Path

import torch
from PIL import Image

from .hy3dgen.rembg import BackgroundRemover
from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from .hy3dgen.texgen import Hunyuan3DPaintPipeline
from pathlib import Path
import uuid
from mmgp import offload, profile_type
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# Globals (lazy-loaded models)
_hunyuan_shape = None
_hunyuan_texture = None
_rembg = None


# -------------------------
# Device Selection
# -------------------------
def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.version.hip is not None:  # ROCm (AMD)
        return "hip"
    elif torch.backends.mps.is_available():
        return "mps"
    elif torch.backends.mps.is_built():  # fallback in case of MPS build but not available
        return "mps"
    elif torch.backends.opencl.is_available():  # not always present, but check
        return "opencl"
    elif torch.has_mps:  # just in case for Apple
        return "mps"
    else:
        return "cpu"


# -------------------------
# Lazy Loader for Rembg
# -------------------------
def get_rembg():
    global _rembg
    if _rembg is None:
        print("[Hunyuan] Loading background remover...")
        _rembg = BackgroundRemover()
    return _rembg


# -------------------------
# Lazy Loader for Hunyuan 3D
# -------------------------
def get_hunyuan_shape_model():
    global _hunyuan_shape

    if _hunyuan_shape is None:
        device = get_best_device()
        print(f"[Hunyuan] Loading Hunyuan 3D Shape Model on {device}...")

        _hunyuan_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mini',
            subfolder='hunyuan3d-dit-v2-mini',
            variant='fp16'
        )
        # _hunyuan_shape.enable_flashvdm(topk_mode='merge')
        print("[Hunyuan] Finished Loading Hunyuan 3D Shape Model")
        # print("[Hunyuan] Offloading Hunyuan 3D Shape Model for MMGP")
        # offload.profile(_hunyuan_shape, profile_type.LowRAM_LowVRAM)
        # print("[Hunyuan] Finished Offloading Hunyuan 3D Shape Model for MMGP")
    return _hunyuan_shape


def get_hunyuan_texture_model():
    global _hunyuan_texture

    if _hunyuan_texture is None:
        print("[Hunyuan] Loading Hunyuan Texture Model...")
        _hunyuan_texture = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        ).to("cpu")
        print("[Hunyuan] Finished loading Hunyuan Texture Model")
        print("[Hunyuan] Offloading Hunyuan Texture Model for MMGP")
        offload.profile(_hunyuan_texture, profile_type.VerylowRAM_LowVRAM )
        print("[Hunyuan] Finished Offloading Hunyuan Texture Model for MMGP")
        
    return _hunyuan_texture


# -------------------------
# Main Function: image → GLB (base64)
# -------------------------
def generate_3d_object_from_image_base64(image_b64: str) -> str:
    """
    Takes a base64 PNG/JPG image
    → removes background (if needed)
    → creates 3D mesh (GLB)
    → textures it
    → returns base64 GLB
    """

    # Convert b64 → PIL image
    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Background removal if needed
    if image.mode == "RGB":
        print("[Hunyuan] Running Rembg...")
        rembg = get_rembg()
        image = rembg(image)

    # Load models
    shape_model = get_hunyuan_shape_model()
    

    # Run Hunyuan 3D Shape Model
    print("[Hunyuan] Generating 3D shape…")
    start_time = time.time()

    mesh = shape_model(
        image=image,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(42),
        output_type="trimesh",
    )[0]

    print(f"[Hunyuan] Shape gen took {time.time() - start_time:.2f} sec")

    # Run Texture Model
    texture_model = get_hunyuan_texture_model()
    print("[Hunyuan] Generating texture…")
    start_time = time.time()
    mesh = texture_model(mesh, image=image)
    print(f"[Hunyuan] Texture gen took {time.time() - start_time:.2f} sec")
    # Export GLB → bytes
    glb_bytes = mesh.export(file_type="glb")
    ensure_dir("output/objects")
    filename = f"hunyuan_{uuid.uuid4().hex}.glb"
    filepath = Path("output/objects") / filename
    with open(filepath, "wb") as f:
        f.write(glb_bytes)
    print(f"[Hunyuan] Saved 3D model to {filepath}")
    
    return base64.b64encode(glb_bytes).decode("utf-8")

    
