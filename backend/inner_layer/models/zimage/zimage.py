import io
import base64
import torch
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from pathlib import Path
import uuid

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# Global variable to store the loaded model
_zimage_pipe = None


def get_zimage_pipeline():
    """
    Lazily load and return the global Z-Image-Turbo pipeline.
    This prevents reloading the model every time the API is called.
    """
    global _zimage_pipe
    
    if _zimage_pipe is None:
        device = get_best_device()
        print(f"Using device: {device}")
        print(f"[ZImage] Loading Z-Image-Turbo on {device}...")
        
        # Use bfloat16 for better quality
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

        _zimage_pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            _zimage_pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            _zimage_pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            _zimage_pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

        _zimage_pipe.enable_model_cpu_offload()

        _zimage_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            _zimage_pipe.scheduler.config,
            use_beta_sigmas=True,
        )
        _zimage_pipe.to(device)
        _zimage_pipe.enable_attention_slicing()

        if hasattr(_zimage_pipe, "enable_vae_slicing"):
            _zimage_pipe.enable_vae_slicing()
            print("VAE slicing enabled")

        if hasattr(getattr(_zimage_pipe, "vae", None), "enable_tiling"):
            _zimage_pipe.vae.enable_tiling()
            print("VAE tiling enabled")
        print("[ZImage] Model loaded successfully.")
    return _zimage_pipe

import torch

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




def generate_image_base64(prompt: str, height: int = 256, width: int = 256, steps: int = 7, seed: int = 5) -> str:
    """
    Generate an image from text prompt using Z-Image-Turbo.
    Returns base64-encoded PNG.
    """
    pipe = get_zimage_pipeline()

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Generating with seed {seed}...")
    
    device = get_best_device()

    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )

    image = result.images[0]

    ensure_dir("output/images")
    filename = f"zimage_{uuid.uuid4().hex}.png"
    filepath = Path("output/images") / filename
    image.save(filepath)
    print(f"[ZImage] Saved image to {filepath}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
