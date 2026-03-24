import base64
import binascii
import io
import os
from pathlib import Path
from PIL import Image
from inner_layer.models.zimage.zimage import generate_image_base64
from inner_layer.models.hunyuan3D.hunyuan3d import generate_3d_object_from_image_base64

def _save_base64_jpg(image_base64: str, output_dir: str | None = None) -> str:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    print("[backend] _save_base64_jpg: received base64 len=", len(image_base64))

    raw = image_base64.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    raw = raw.replace("\n", "").replace("\r", "").replace(" ", "")
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad

    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        print("[backend] _save_base64_jpg: base64 decode failed")
        raise ValueError("Invalid base64 image data") from exc
    if not data:
        raise ValueError("Decoded image data is empty")

    print("[backend] _save_base64_jpg: decoded bytes=", len(data))
    backend_root = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = backend_root / "inputs" / "images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"input_{os.urandom(4).hex()}.jpg"
    output_path = output_dir / filename

    try:
        img = Image.open(io.BytesIO(data))
        print("[backend] _save_base64_jpg: image format=", img.format, "size=", img.size, "mode=", img.mode)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(output_path, format="JPEG", quality=95)
        print("[backend] _save_base64_jpg: saved to", output_path)
    except Exception as exc:
        print("[backend] _save_base64_jpg: image decode/save failed:", exc)
        raise ValueError("Decoded data is not a valid image") from exc

    return str(output_path)

def generate_image_from_text(text: str) -> str:
    return generate_image_base64(text, convert_to_glb=True)

def generate_object_from_image(image_base64: str) -> str:
    print("[backend] generate_object_from_image: start")
    image_path = _save_base64_jpg(image_base64)
    try:
        backend_root = Path(__file__).resolve().parents[1]
        rel_path = Path(image_path).relative_to(backend_root)
    except ValueError:
        rel_path = image_path
    print("[backend] generate_object_from_image: saved image at", rel_path)
    print("[backend] generate_object_from_image: calling Hunyuan3D with", image_path)
    return generate_3d_object_from_image_base64(image_path)

def generate_object_from_text(text: str) -> str:
    image_path = generate_image_base64(text, convert_to_glb=False)
    object_result = generate_3d_object_from_image_base64(image_path)
    return object_result
