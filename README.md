# VR-Image-3D-Object-Generator
Libraries Installation and Start Backend:
```bash
cd ./backend

# basic libraries
pip install -r ./requirements.txt

# for object generation
cd ./inner_layer/models/hunyuan3D/
pip install -r ./hunyuan3d_requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install

# starting backend server
cd ../../../../../../
uvicorn outer_layer.app:app --reload
```
