# VR-Image-3D-Object-Generator
Dependencies Installation and Start Backend:
```bash
cd ./backend

python3 -m venv venv
source ./venv/bin/activate

# installing basic dependencies
pip install -r ./requirements.txt

# installing dependencies for object generation
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

Dependencies Installation and Start Frontend:
```bash
cd ./frontend

# installing dependencies
npm install

# generating openssl keys for https
openssl req -nodes -new -x509 -keyout key.pem -out cert.pem -days 365

# starting frontend server
node server.js
```
