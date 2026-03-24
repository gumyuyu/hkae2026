import base64
import os
import shutil
from pathlib import Path
from gradio_client import Client, handle_file

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def generate_3d_object_from_image_base64(image_path: str) -> str:
    """
    Converts a base64 image string to PNG, generates a 3D object using Hunyuan3D,
    saves the textured mesh (.glb) in output/shapes/, and returns its base64 string.
    """
    print("[hunyuan3d] input image path:", image_path)
    print("[hunyuan3d] cwd:", Path.cwd())
    input_path = Path(image_path).resolve()
    if not input_path.exists():
        print("[hunyuan3d] input image exists: False")   
        raise ValueError(f"Input image does not exist: {input_path}")
    print("[hunyuan3d] input image abs:", input_path)
    print("[hunyuan3d] input image size:", input_path.stat().st_size)
    ensure_dir("output/images")
    debug_copy = Path("output/images") / f"hunyuan_input_{input_path.stem}{input_path.suffix}"
    try:
        shutil.copyfile(input_path, debug_copy)
        print("[hunyuan3d] copied input image to:", debug_copy)
    except Exception as exc:
        print("[hunyuan3d] failed to copy input image:", exc)
    # --- Initialize Hunyuan3D client ---
    client = Client("tencent/Hunyuan3D-2.1", token=os.getenv("HF_TOKEN"))

    # --- Predict 3D object ---
    try:
        result = client.predict(
            api_name="/generation_all",
            image=handle_file(image_path),
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=30,
            guidance_scale=5,
            seed=1234,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True
        )
    except Exception as exc:
        print("[hunyuan3d] client.predict failed:", exc)
        raise

    # The generated GLB path returned by Hunyuan3D
    textured_mesh_path = result[1]['value']
    print("[hunyuan3d] textured mesh path:", textured_mesh_path)

    # --- Ensure output directory exists ---
    ensure_dir("output/shapes")
    filename = f"shape_{Path(textured_mesh_path).stem}_{os.urandom(4).hex()}.glb"
    output_path = Path("output/shapes") / filename

    # --- Copy or move the generated GLB to output/shapes ---
    with open(textured_mesh_path, "rb") as src_file, open(output_path, "wb") as dst_file:
        dst_file.write(src_file.read())

    print(f"[3D] Saved GLB to {output_path}")

    # --- Return base64 string of saved GLB ---
    with open(output_path, "rb") as mesh_file:
        mesh_b64 = base64.b64encode(mesh_file.read()).decode("utf-8")

    return mesh_b64
